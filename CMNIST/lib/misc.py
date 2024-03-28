# Code adopted from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/misc.py

import math
import sys
import numpy as np
import torch
from .iro_utils import aggregation_function

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def accuracy(network, loader, device, weights=None, alpha=None):
    correct = 0
    total = 0
    weights_offset = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if alpha is None:
                p = network.predict(x)
            else:
                t_alpha = torch.tile(torch.tensor(alpha),(x.shape[0],1)).to(device)
                p = network.predict(x,t_alpha)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def acc_for_cvar(risks, accs, alpha):
    var = torch.quantile(risks,alpha, interpolation='linear')
    avg_acc = accs[risks >= var].mean()
    return avg_acc


def cvar(network, loaders, loss_fn, device, alphas, not_h=True):
    network.eval()
    aggregator = aggregation_function(name='cvar')
    if not_h:
        risks = []
        with torch.no_grad():
            for loader in loaders:
                loss_value = loss(network, loader, loss_fn, device)
                acc_value = accuracy(network, loader, device) 
                risks.append(loss_value)
        risks = torch.tensor(risks)
        cvars = [aggregator.aggregate(risks, alpha) for alpha in alphas]
        print([round(cvar.item(),2) for cvar in cvars])
        return 
    cvars, acc_cvars = [], []
    for alpha in alphas:
        risks, accs = [], []
        with torch.no_grad():
            for loader in loaders:
                loss_value = loss(network, loader, loss_fn, device, alpha)
                acc_value = accuracy(network, loader, device, alpha=alpha) 
                risks.append(loss_value)
                accs.append(acc_value)
        risks, accs = torch.tensor(risks), torch.tensor(accs)
        cvar = aggregator.aggregate(risks, alpha)
        acc_cvar = acc_for_cvar(risks, accs, alpha)
        cvars.append(round(cvar.item(),2))
        acc_cvars.append(round(acc_cvar.item(),2))
    print(cvars, acc_cvars)
    return


def loss(network, loader, loss_fn, device, alpha=None):
    running_loss = 0
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if alpha is None:
                running_loss += loss_fn(network.predict(x), y).item()
            else:
                t_alpha = torch.tile(torch.tensor(alpha),(x.shape[0],1)).to(device)
                running_loss += loss_fn(network.predict(x, t_alpha), y).item()

    network.train()
    return running_loss / len(loader)


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
