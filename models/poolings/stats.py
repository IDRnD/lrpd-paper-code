import torch
import torch.nn as nn

def mean_std_pooling(x,dim = -1):
    mean = torch.mean(x, dim=dim, keepdim=False)
    std = torch.std(x, dim=dim, unbiased=True, keepdim=False)
    x = torch.cat([mean, std], dim=1)
    return x

def mean_var_pooling(x,dim = -1):
    mean = torch.mean(x, dim=dim,keepdim=False)
    var = torch.var(x, dim=dim, unbiased=True, keepdim=False)
    x = torch.cat([mean, var], dim=1)
    return x

STAT_POOLINGS = {
    "var" : mean_std_pooling,
    "std" : mean_var_pooling
}