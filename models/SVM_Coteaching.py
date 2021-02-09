import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch as torch

class SVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(SVDD, self).__init__()
        self.c1 = torch.zeros(z_dim)
        self.R1 = None

        self.SVM1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim, bias=False)
        )

        self.decoder_pretrain = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, x1):
        z1 = self.SVM1(x1)
        xhat = self.decoder_pretrain(z1)
        return z1, xhat

    def init_c(self, c1):
        self.c1 = c1
        return c1

    def distance(self, z1):
        distance1 = torch.sqrt(((z1 - self.c1) ** 2).sum(dim=1))
        return distance1


class SVMLoss(torch.nn.Module):

    def __init__(self):
        super(SVMLoss, self).__init__()

    def forward(self, z1, c1):
        loss = torch.sqrt(((z1 - c1) ** 2).mean())
        return loss