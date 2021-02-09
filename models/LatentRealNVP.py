import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions
from torch.nn.parameter import Parameter

class Model_LatentRealNVP(nn.Module):
    def __init__(self, nets, nett, encoder, decoder, mask, prior):
        super(Model_LatentRealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.encoder = torch.nn.ModuleList([encoder()])
        self.decoder = torch.nn.ModuleList([decoder()])

    def encode(self, x):
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x

    def decode(self, x):
        for i in range(len(self.encoder)):
            x = self.decoder[i](x)
        return x

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        pz = self.prior.log_prob(z.cpu())
        pz = pz.cuda()
        return pz + logp



    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        z = z.cuda()
        # logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


class CIFAR10_LatentRealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior, rep_dim=128):
        super(CIFAR10_LatentRealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.pool = nn.MaxPool2d(2, 2)
        self.rep_dim = rep_dim

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=True, padding=2)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=True, padding=2)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 5, bias=True, padding=2)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=True)
        self.fc1 = nn.Linear(128 * 4 * 4, self.rep_dim, bias=True)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=True)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), 128, 5, bias=True, padding=2)
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, bias=True, padding=2)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=True)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, bias=True, padding=2)
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=True)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 5, bias=True, padding=2)



    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        # x = self.pool(F.leaky_relu(x))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        # x = self.pool(F.leaky_relu(x))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        # x = self.pool(F.leaky_relu(x))
        x = x.view(x.size(0), -1)
        z = self.bn1d(self.fc1(x))
        # z = self.fc1(x)
        return z

    def decode(self, z):
        x = z.view(z.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x)), scale_factor=2)
        # x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x)), scale_factor=2)
        # x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x)), scale_factor=2)
        # x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv4(x)
        return x

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        pz = self.prior.log_prob(z.cpu())
        pz = pz.cuda()
        return pz + logp

    # def log_prob(self, x):
    #     z, logp = self.f(x)
    #     pz = self.prior.log_prob(z.cpu())
    #     pz = pz.cuda()
    #     return pz + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x




class MNIST_LantetRealNVP(nn.Module):

    def __init__(self, nets, nett, mask, prior, rep_dim=32):
        super(MNIST_LantetRealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])
        self.pool = nn.MaxPool2d(2, 2)
        self.rep_dim = rep_dim

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encode(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        return z

    def decode(self, z):
        x = z.view(z.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        pz = self.prior.log_prob(z.cpu())
        pz = pz.cuda()
        return pz + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x




class MCI_LatentRealNVP(nn.Module):
    def __init__(self, nets, nett, input_dim, hidden_dim, z_dim, mask, prior):
        super(MCI_LatentRealNVP, self).__init__()

        self.z_dim = z_dim
        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))])


        self.lstm_encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lstm_encoder_bottleneck = nn.LSTM(input_size=hidden_dim, hidden_size=z_dim, batch_first=True)
        self.fcn_bottleneck1 = nn.Linear(z_dim * 7, z_dim)
        self.tanh = nn.Tanh()
        self.leaklyrelu2 = nn.LeakyReLU()
        self.fcn_bottleneck2 = nn.Linear(z_dim, z_dim * 7)
        self.lstm_decoder_bottleneck = nn.LSTM(input_size=z_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.lstm_decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim, batch_first=True)

    def encode(self, x):
        z, _ = self.lstm_encoder(x)
        z, (h, ) = self.lstm_encoder_bottleneck(z)
        return h

    def decode(self, z):
        # x = self.fcn_bottleneck2(z)
        # x = x.reshape(x.shape[0], 7, self.z_dim)
        # x, _ = self.lstm_decoder_bottleneck(x)
        # x, _ = self.lstm_decoder(x)
        z = z.unsqueeze(1)
        z = z.repeat(1, 7, 1)
        x = self.lstm_decoder(z)
        return x

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        pz = self.prior.log_prob(z.cpu())
        pz = pz.cuda()
        return pz + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        z = z.cuda()
        # logp = self.prior.log_prob(z)
        x = self.g(z)
        return x
