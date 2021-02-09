import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch as torch


class AE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(AE, self).__init__()
        self.z_dim = z_dim
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu1_layer = nn.Linear(hidden_dim, z_dim)
        self.logvar1_layer = nn.Linear(hidden_dim, z_dim)

        self.decoder1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu2_layer = nn.Linear(hidden_dim, z_dim)
        self.logvar2_layer = nn.Linear(hidden_dim, z_dim)

        self.decoder2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x1, x2):
        # a, b, c = x_nn1.shape
        enc1 = self.encoder1(x1)
        mu1 = self.mu1_layer(enc1)
        logvar1 = self.logvar1_layer(enc1)
        z1 = self.reparameterize(mu1, logvar1)
        xhat1 = self.decoder1(z1)
        enc2 = self.encoder2(x2)
        mu2 = self.mu2_layer(enc2)
        logvar2 = self.logvar2_layer(enc2)
        z2 = self.reparameterize(mu2, logvar2)
        xhat2 = self.decoder2(z2)
        return z1, z2, xhat1, xhat2, mu1, mu2, logvar1, logvar2




