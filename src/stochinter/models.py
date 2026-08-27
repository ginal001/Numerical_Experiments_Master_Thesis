import torch
from torch import nn


class UnconditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class ConditionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, t, omega):
        return self.net(torch.cat([x, t, omega], dim=-1))


class PotentialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x, t):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            phi = self.net(torch.cat([x, t], dim=-1))
            v = torch.autograd.grad(
                outputs=phi, 
                inputs=x, 
                grad_outputs=torch.ones_like(phi),
                create_graph=True
            )[0]
        return v


class DirectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))
