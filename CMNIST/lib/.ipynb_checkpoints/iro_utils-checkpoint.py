import torch
import torch.distributions as dist
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from .kde import Nonparametric
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
        elif self.name == 'cvar-dist':
            return self.cvar_dist(risks, alpha)
        else:
            raise NotImplementedError("Currently, only CVaR is implemented.")
    
    def cvar_full(self, risks, alpha) -> float:
        var = Quantile.apply(risks, alpha)
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
    
    def cvar_dist(self, risks, alpha) -> float:
        if alpha>=0 and alpha<=0.2: return self.cvar(risks, alpha)
        dist = Nonparametric()
        dist.estimate_parameters(risks)
        if alpha>=0.95 and alpha<=1: return dist.icdf(-1000)
        obser = torch.arange(alpha,1.0,10)
        cvar = torch.stack([dist.icdf(percentile) for percentile in obser]).mean()
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
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.aggregator = aggregation_function(name="cvar-diff")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.a = torch.tensor([0.5], requires_grad=True, device=self.device, dtype=torch.float32)
        self.b = torch.tensor([0.5], requires_grad=True, device=self.device, dtype=torch.float32)
        self.optimizer_dist = torch.optim.Adam([self.a, self.b], lr=1e-5)
    
    def aggregated_objective(self, model, minibatches, num_samples=5):
        ### reparameterization needed here.
        uniform_samples = dist.Uniform(0,1).sample((num_samples,))
        uniform_samples.requires_grad=True
        alphas = []
        for t_unif in uniform_samples:
            alphas.append(IcdfBetaScaler.apply(t_unif, self.a, self.b))
        cvar_estimates = []
        for alpha in alphas:
            risks = []
            for x, y in minibatches:
                t_alpha = torch.tile(alpha,(x.shape[0],1))
                risks.append(self.loss_fn(model(x,t_alpha), y).reshape(1))
            risks = torch.cat(risks).detach()
            cvar_estimates.append(self.aggregator.aggregate(risks, alpha))
        cvar_estimates = torch.stack(cvar_estimates)
        average_cvar = torch.mean(cvar_estimates)
        return average_cvar
    
    def update(self, model, minibatches):
        for param in model.parameters():
            param.requires_grad = False
        avg_cvar = self.aggregated_objective(model, minibatches)
        avg_cvar.backward()
        self.optimizer_dist.step()
        self.optimizer_dist.zero_grad()
        return self.a.detach().item(), self.b.detach().item()
