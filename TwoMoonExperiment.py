import numpy as np
import matplotlib.pyplot as plt

from K_spectral_norm import spectral_norm
from pylab import rcParams
rcParams['figure.figsize'] = 16, 9
rcParams['figure.dpi'] = 300
import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False).cuda()
        self.t = torch.nn.ModuleList([nett() for _ in range(len(mask))]).cuda()
        self.s = torch.nn.ModuleList([nets() for _ in range(len(mask))]).cuda()

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
        # log_prob_z = self.prior.log_prob(z.data.cpu().numpy())
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


nets = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2), nn.Tanh())
nett = lambda: nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(), nn.Linear(256, 256), nn.LeakyReLU(), nn.Linear(256, 2))
masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
prior = distributions.MultivariateNormal(torch.zeros(2).cuda(), torch.eye(2).cuda())

training_contamination = 0.05
batch_size = 128
torch.manual_seed(0)
np.random.seed(0)
torch.set_deterministic(True)
n_sample = 10000

moons_data = datasets.make_moons(n_samples=10000, noise=.1)[0].astype(np.float32)
# outlier = (np.random.rand(int(n_sample * training_contamination), 2)-0.35) * 5  # uniform
outlier = (np.random.normal(loc=[2.5, 2.5], scale=0.05, size=[int(n_sample * training_contamination), 2]))  # concentrated
noisy_moons_all = np.concatenate([moons_data, outlier], axis=0).astype(np.float32)
min_x, max_x = min(noisy_moons_all[:, 0]), max(noisy_moons_all[:, 0])
min_y, max_y = min(noisy_moons_all[:, 1]), max(noisy_moons_all[:, 1])

label = np.ones(noisy_moons_all.shape[0])
label[:n_sample] = 0

plt.subplot(141)
plt.axis('off')
with sns.axes_style("darkgrid"):
    sns.scatterplot(noisy_moons_all[:, 0], noisy_moons_all[:, 1], hue=label, style=label, palette='Set2')
    plt.legend(fontsize='x-large', title_fontsize='200')
figure_layout = [142, 143, 144]
figure_index = 0


rho = 0.05
L_hat = 100
for method in ['naive', 'trim', 'spectral']:
    flow = RealNVP(nets, nett, masks, prior)
    flow = flow.cuda()
    if method == 'spectral':
        print("adding spectral norm")
        for block in flow.s:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    spectral_norm(layer, L=1)

        for block in flow.t:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    spectral_norm(layer, L=1)

    optimizer = torch.optim.Adam(flow.parameters(), lr=3e-4)
    for t in range(5001):
        sample_index = np.random.choice(noisy_moons_all.shape[0], batch_size, replace=False)
        noisy_moons = noisy_moons_all[sample_index, :]
        flow.zero_grad()
        if method == 'trim':
            negative_log_density = -flow.log_prob(torch.from_numpy(noisy_moons).float().cuda())
            with torch.no_grad():
                _, index_loss = torch.sort(negative_log_density)
                index_loss = index_loss[:int(batch_size * (1 - training_contamination))]

            negative_log_density = negative_log_density[index_loss].mean()
            negative_log_density.backward()
            optimizer.step()

        elif method == 'spectral':
            negative_log_density = -flow.log_prob(torch.from_numpy(noisy_moons).float().cuda())
            with torch.no_grad():
                _, index_loss = torch.sort(negative_log_density)
                index_loss = index_loss[:int(batch_size * (1 - training_contamination))]
            negative_log_density = negative_log_density[index_loss].mean()
            negative_log_density.backward()
            optimizer.step()

        elif method == 'naive':
            flow.zero_grad()
            negative_log_density = -flow.log_prob(torch.from_numpy(noisy_moons).float().cuda())
            negative_log_density = negative_log_density.mean()
            negative_log_density.backward()
            optimizer.step()

        if t % 50 == 0:
            print('iter %s:' % t, 'loss density = %.3f' % negative_log_density.mean())

    flow.eval()
    uniform_x = np.linspace(min_x, max_x, 5000)
    uniform_y = np.linspace(min_y, max_y, 5000)
    xx, yy = np.meshgrid(uniform_x, uniform_y)
    uniform_data = np.stack([xx, yy])


    density_map = []
    with torch.no_grad():
        for i in range(5000):
            log_prob = flow.log_prob(torch.from_numpy(uniform_data[:, i, :].transpose()).cuda().float())
            density_map.append(np.exp(log_prob.data.cpu().numpy()))
    density_map = np.stack(density_map)
    plt.subplot(figure_layout[figure_index])
    plt.axis('off')

    minmax_scaler = MinMaxScaler()
    density_map = minmax_scaler.fit_transform(density_map)
    plt.contourf(uniform_x, uniform_y, density_map, 20, cmap=sns.color_palette("viridis", as_cmap=True))
    if figure_index == 2:
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
    figure_index += 1

plt.figure(figsize=(24, 20))
plt.tight_layout()
plt.show()



