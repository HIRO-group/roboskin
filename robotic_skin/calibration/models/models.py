import torch
import torch.nn as nn


class DHNet(nn.Module):
    """
    network used to predict dh parameters
    """
    def __init__(self, num_params, objective_fnc):
        self.params = nn.Parameter(torch.randn(num_params), requires_grad=True)
        self.objective_fnc = objective_fnc

    def forward(self):
        out = self.objective_fnc(self.params)
        return out
