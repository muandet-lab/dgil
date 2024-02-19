import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from numpy.polynomial.chebyshev import Chebyshev
from scipy.stats import beta
import copy


class Quantile(torch.autograd.Function):

    @staticmethod
    def forward(ctx, risks, alpha):
        q = torch.quantile(risks, alpha)
        ctx.save_for_backward(risks, alpha, q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        risks, alpha, q = ctx.saved_tensors
        diff = 1e-5
        grad_risks = grad_output * (risks == q).int()
        if alpha < diff:
            local_grad_alpha = (torch.quantile(risks, alpha+diff) - torch.quantile(risks, alpha))/diff
        elif alpha + diff > 1.0:
            local_grad_alpha = (torch.quantile(risks, alpha) - torch.quantile(risks, alpha-diff))/diff
        else:
            local_grad_alpha = (torch.quantile(risks, alpha+diff) - torch.quantile(risks, alpha-diff)) / (2.0 * diff)
        grad_alpha = grad_output * local_grad_alpha
        return grad_risks, grad_alpha

class aggregation_function:    
    """ This class aggregates the risks. """
    def __init__(self, name:str):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def aggregate(self, risks, alpha) -> float:
        if self.name == 'cvar':
            return self.cvar(risks, alpha)
        elif self.name == 'cvar-full':
            return self.cvar_full(risks, alpha)
        elif self.name == 'cvar-diff':
            return self.cvar_diff(risks, alpha)
        else:
            raise NotImplementedError("Currently, only CVaR is implemented.")
    
    def cvar_full(self, risks, alpha) -> float:
        var = Quantile.apply(risks, torch.tensor(alpha,device=self.device))
        cvar_plus = risks[risks >= var].mean()
        lambda_alpha = ((risks <= var).sum().div(len(risks)) - alpha) / (1 - alpha)
        cvar = lambda_alpha * var + (1 - lambda_alpha) * cvar_plus
        return cvar
    
    def cvar_diff(self, risks, base_alpha) -> float:
        number_of_points = 5
        alphas = [(1-base_alpha)*(i/number_of_points)+base_alpha for i in range(number_of_points)]
        quantiles = torch.stack([Quantile.apply(risks, alpha) for alpha in alphas])
        return quantiles.mean()
    
    def cvar(self, risks, alpha) -> float:
        var = torch.quantile(risks,alpha, interpolation='linear')
        cvar = risks[risks >= var].mean()
        return cvar

class IcdfBetaScaler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a, b):
        ctx.save_for_backward(x, a, b)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.tensor(beta.ppf(x.item(), a.item(), b.item())).float().to(device)

    @staticmethod
    def backward(ctx, grad_output):
        x, a, b = ctx.saved_tensors
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        diff = 1e-5
        x = x.item()
        a = a.item()
        b = b.item()
        if x < diff:
            local_grad_x = torch.tensor([(beta.ppf(x+diff, a, b) - beta.ppf(x, a, b)) / diff]).float()
        elif x + diff > 1.0:
            local_grad_x = torch.tensor([(beta.ppf(x, a, b) - beta.ppf(x-diff, a, b)) / diff]).float()
        else:
            local_grad_x = torch.tensor([(beta.ppf(x+diff, a, b) - beta.ppf(x-diff, a, b)) / (2.0 * diff)]).float()

        if a < diff:
            local_grad_a = torch.tensor([(beta.ppf(x, a+diff, b) - beta.ppf(x, a, b)) / diff]).float()
        else:
            local_grad_a = torch.tensor([(beta.ppf(x, a+diff, b) - beta.ppf(x, a-diff, b)) / (2.0 * diff)]).float()

        if b < diff:
            local_grad_b = torch.tensor([(beta.ppf(x, a, b+diff) - beta.ppf(x, a, b)) / diff]).float()
        else:
            local_grad_b = torch.tensor([(beta.ppf(x, a, b+diff) - beta.ppf(x, a, b-diff)) / (2.0 * diff)]).float()
        grad_x = grad_output * local_grad_a.to(device)
        grad_a = grad_output * local_grad_a.to(device)
        grad_b = grad_output * local_grad_b.to(device)
        return grad_x, grad_a, grad_b

class Pareto_distribution:
    def __init__(self, env_dict):
        self.env_dict = env_dict
        self.loss_fn = torch.nn.MSELoss()
        self.aggregator = aggregation_function(name="cvar-diff")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def compute_norm(self, model):
        # Calculate norm of gradients
        total_norm = 0
        for param in model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm**2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def aggregated_objective(self, model, a, b, num_samples=5):
        ### reparameterization needed here.
        uniform_samples = dist.Uniform(0,1).sample((num_samples,))
        alphas = []
        for unif_sample in uniform_samples:
            t_unif = torch.tensor(unif_sample, requires_grad=True, device=self.device)
            alphas.append(IcdfBetaScaler.apply(t_unif, a, b))
        cvar_estimates = []
        for alpha in alphas:
            risks = []
            for e in self.env_dict.keys():
                x, y = self.env_dict[e]['x'].to(self.device), self.env_dict[e]['y'].to(self.device) 
                x.requires_grad, y.requires_grad = False, False
                risks.append(self.loss_fn(y, model(x,alpha)))
            risks = torch.stack(risks).detach()
            cvar_estimates.append(self.aggregator.aggregate(risks, alpha))
        cvar_estimates = torch.stack(cvar_estimates)
        average_cvar = torch.mean(cvar_estimates)
        return average_cvar
    
    def optimize(self, model):
        for param in model.parameters():
            param.requires_grad = False
        a = torch.tensor([1.0], requires_grad=True, device=self.device, dtype=torch.float32)
        b = torch.tensor([1.0], requires_grad=True, device=self.device, dtype=torch.float32)
        optimizer_dist = torch.optim.Adam([a, b], lr=0.01)
        num_epochs = 10
        for epoch in range(num_epochs):
            avg_cvar = self.aggregated_objective(model, a, b)
            avg_cvar.backward()
            optimizer_dist.step()
            optimizer_dist.zero_grad()
        return a.detach().item(), b.detach().item()

class ARM_Regression:
    def __init__(self, name, experiment="1D_linear"):      
        self.aggregator = aggregation_function(name=name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_cvar_h(self, alpha, h, env_dict):
        loss_fn = torch.nn.MSELoss()        
        risks = []
        for e in env_dict.keys():
            output = h(env_dict[e]['x'].to(self.device), alpha.to(self.device))
            risks.append(loss_fn(output,env_dict[e]['y'].to(self.device)))
        risks = torch.stack(risks)
        cvar = self.aggregator.aggregate(risks, alpha)
        return cvar
    
    def fit_h(self, h, env_dict, a, b, num_epochs=30):
        loss_fn = torch.nn.MSELoss()
        alphas = np.random.beta(a=a, b=b, size=5)
        alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)
        learning_rate = 0.1
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        for epoch in range(num_epochs):
            avg_cvar = torch.mean(torch.stack([self.compute_cvar_h(alpha, h, env_dict) for alpha in alphas]))
            avg_cvar.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_cvar.item()}")
        return 
    
    def fit_f(self, f, env_dict, alpha):        
        learning_rate = 0.1
        num_epochs= 100
        loss_fn = torch.nn.MSELoss()        
        optimizer = torch.optim.Adam(f.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            risks = []
            for e in env_dict.keys():
                x, y = env_dict[e]['x'].to(self.device), env_dict[e]['y'].to(self.device) 
                risks.append(loss_fn(f(x),y))
            risks = torch.stack(risks)
            cvar = self.aggregator.aggregate(risks, alpha)
            cvar.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {cvar.item()}")
        return 
    
    def fit_h_as_f(self, h, env_dict, alpha): 
        t_alpha = torch.tensor(alpha).to(self.device)
        learning_rate = 0.1
        num_epochs= 100
        loss_fn = torch.nn.MSELoss()        
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
        for epoch in range(num_epochs):
            risks = []
            for e in env_dict.keys():
                x, y = env_dict[e]['x'].to(self.device), env_dict[e]['y'].to(self.device) 
                risks.append(loss_fn(h(x,t_alpha),y))
            risks = torch.stack(risks)
            cvar = self.aggregator.aggregate(risks, alpha)
            cvar.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {cvar.item()}")
        return
    
    def fit_h_pareto(self, h, env_dict, num_epochs=30):
        loss_fn = torch.nn.MSELoss()
        learning_rate = 0.1
        optimizer = torch.optim.Adam(h.parameters(), lr=learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        p_min = Pareto_distribution(env_dict)
        for epoch in range(num_epochs):
            a, b = p_min.optimize(copy.deepcopy(h))
            alphas = np.random.beta(a, b, size=5)
            alphas = torch.tensor(alphas, dtype=torch.float32).to(self.device)
            avg_cvar = torch.mean(torch.stack([self.compute_cvar_h(alpha, h, env_dict) for alpha in alphas]))
            avg_cvar.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_cvar.item()}")
        return  
