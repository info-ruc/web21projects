import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


def calc_coeff(iter_num, high=0.1, low=0.0, alpha=1.0, max_iter=1000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(- alpha * iter_num / max_iter)) - (high - low) + low)


class relation_module(nn.Module):
    def __init__(self, in_dim, width, class_num, grl=False):
        super(relation_module, self).__init__()
        self.grl = grl

        self.fc1 = nn.Linear(in_dim, width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(width, 64)
        self.fc3 = nn.Linear(64, class_num)

        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0.0)
        self.fc3.weight.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0.0)

        if self.grl:
            self.iter_num = 0
            self.alpha = 1.0
            self.low = 0.0
            self.high = 0.1
            self.max_iter = 1000.0

    def forward(self, x):
        if self.grl:
            if self.training:
                self.iter_num += 1
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
            x = x * 1.0
            x.register_hook(grl_hook(coeff))
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)

        return self.fc3(x)


class AdvNet(nn.Module):
    def __init__(self, in_dim=640, use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=10):
        super(AdvNet, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.bottleneck_layer = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        self.relation_module = relation_module(bottleneck_dim, width, class_num, False)
        
        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def forward(self, x):
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)

        return self.relation_module(x)

    def get_parameter_list(self):
        parameter_list = [
            {"params":self.bottleneck_layer.parameters(), "lr":1},
            {"params":self.relation_module.parameters(), "lr":1}
        ]
        return parameter_list