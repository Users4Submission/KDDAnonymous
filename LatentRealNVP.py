from models.LatentRealNVP import Model_LatentRealNVP
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import torch as torch
import argparse
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch import distributions
import os as os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support as prf,
    accuracy_score,
    roc_auc_score,
)
import numpy as np
from utils import *
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.optim import Adam
from data_process import RealDataset, RealGraphDataset
from K_spectral_norm import spectral_norm

class Solver_LatentRealNVP():
    def __init__(self, data_name, hidden_dim=256, seed=0, z_dim=10, learning_rate=3e-4,
                 batch_size=128, training_ratio=0.8, max_epochs=100):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.result_path = "./results/{}/0.0/LatentRealNVP/{}/".format(
            data_name, seed
        )
        data_path = "./data/" + data_name + ".npy"
        self.model_save_path = "./trained_model/{}/LatentRealNVP/{}/".format(data_name, seed)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.z_dim = z_dim

        # self.dataset = RealGraphDataset(data_path, missing_ratio=0, radius=2)
        self.dataset = RealDataset(data_path, missing_ratio=0)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs

        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * training_ratio)
        self.n_test = n_sample - self.n_train
        print('|data dimension: {}|data noise ratio:{}'.format(self.dataset.__dim__(), self.data_anomaly_ratio))

        training_data, testing_data = data.random_split(dataset=self.dataset,
                                                                         lengths=[
                                                                             self.n_train,
                                                                             self.n_test
                                                                         ])

        self.training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
        self.testing_loader = data.DataLoader(testing_data, batch_size=self.n_test, shuffle=False)
        self.ae = None
        self.build_model()
        self.print_network()

    def build_model(self):
        nets = lambda: nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU(),
                                     nn.Linear(self.hidden_dim, self.z_dim), nn.Tanh())
        nett = lambda: nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU(), nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU(),
                                     nn.Linear(self.hidden_dim, self.z_dim))

        encoder = lambda: nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5),
                                        nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5),
                                        nn.Linear(self.hidden_dim, self.z_dim), nn.Tanh())
        decoder = lambda: nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5),
                                        nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU(), nn.Dropout(0.5),
                                        nn.Linear(self.hidden_dim, self.input_dim), nn.Sigmoid())

        first_mask = np.array([0] * self.z_dim)
        second_mask = np.array([0] * self.z_dim)
        first_mask[int(self.z_dim/2):] = 1
        second_mask[:(self.z_dim - int(self.z_dim/2))] = 1

        masks = torch.from_numpy(np.array([first_mask, second_mask] * 3).astype(np.float32))  # 3 is the number of layers
        prior = distributions.MultivariateNormal(torch.zeros(self.z_dim), torch.eye(self.z_dim))
        self.ae = Model_LatentRealNVP(nets, nett, encoder, decoder, masks, prior)
        self.ae = self.ae.cuda()

    def print_network(self):
        num_params = 0
        for p in self.ae.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def pretrain(self):
        optimizer_ae = Adam(self.ae.parameters(), lr=self.learning_rate)
        self.ae.train()
        training_contamination = self.data_anomaly_ratio
        for epoch in range(200):
            training_loss = 0.0
            for batch_idx, (x, _, _) in enumerate((self.training_loader)):
                """ train RealNVP"""
                x = to_var(x)
                x = x.float()
                optimizer_ae.zero_grad()
                self.ae.zero_grad()
                z = self.ae.encode(x)
                xhat = self.ae.decode(z)
                loss = ((xhat - x) ** 2).sum(axis=1)
                _, index = torch.sort(loss)
                loss = loss[index[:int(self.batch_size * (1 - training_contamination))]].mean()
                loss.backward()
                optimizer_ae.step()


    def train(self):
        optimizer_ae = Adam(self.ae.parameters(), lr=self.learning_rate)
        self.ae.train()

        training_contamination = self.data_anomaly_ratio
        for block in self.ae.s:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    # nn.utils.spectral_norm(layer)
                    spectral_norm(layer, L=1)

        for block in self.ae.t:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    # nn.utils.spectral_norm(layer)
                    spectral_norm(layer, L=1)
        for epoch in tqdm(range(self.max_epochs)):
            training_loss = 0.0
            for batch_idx, (x, _, _) in enumerate((self.training_loader)):
                """ train RealNVP"""
                x = to_var(x)
                x = x.float()
                # fake_z = self.ae.sample(self.batch_size)
                # fake_x = self.ae.decode(fake_z)

                # x = torch.cat([x, fake_x.squeeze()], dim=0)
                optimizer_ae.zero_grad()
                self.ae.zero_grad()
                z = self.ae.encode(x)
                xhat = self.ae.decode(z)
                reconstruction_loss = ((xhat - x) ** 2).sum(axis=1)
                _, index = torch.sort(reconstruction_loss)
                reconstruction_loss = 0.01*reconstruction_loss[index[:int(self.batch_size * (1 - training_contamination))]].mean()
                reconstruction_loss.backward(retain_graph=True)
                loss = -self.ae.log_prob(z)
                # loss = loss.mean()
                _, index = torch.sort(loss)
                loss = loss[index[:int(self.batch_size * (1 - training_contamination))]].mean()
                loss.backward()
                # training_loss = training_loss + loss.item()
                optimizer_ae.step()

            # print("training epoch: {}| training loss: {:0.3f}".format(epoch, training_loss))
            # self.test()

    def test(self):
        log_density_test = []
        y_test = []

        self.ae.eval()
        for batch_idx, (x, y, _) in enumerate(self.testing_loader):
            x = to_var(x)
            x = x.float()
            y = y.float()
            log_density = self.ae.log_prob(self.ae.encode(x))
            y_test.append(y)

            log_density_test.append(log_density)

        log_density_test = torch.cat(log_density_test)
        y_test = torch.cat(y_test)

        y_test = y_test.data.cpu().numpy()
        log_density_test = log_density_test.data.cpu().numpy()

        clean_index = np.where(y_test.squeeze() == 0)
        anomaly_index = np.where(y_test.squeeze() == 1)

        thresh = np.percentile(log_density_test, (1 - self.data_normaly_ratio) * 100)
        print("Threshold :", thresh)

        pred = (log_density_test < thresh).astype(int)
        gt = y_test.astype(int)
        auc = roc_auc_score(gt, -log_density_test)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC:{:0.4f}".format(
            accuracy, precision, recall, f_score, auc))

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
                "auc": auc,
            },
        )
        return accuracy, precision, recall, f_score, auc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument(
        "--seed", type=int, default=5, required=False
    )
    parser.add_argument(
        "--data", type=str, default="thyroid", required=False
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, required=False
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, required=False
    )
    parser.add_argument(
        "--round_D", type=int, default=1, required=False
    )
    parser.add_argument(
        "--z_dim", type=int, default=10, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, required=False
    )
    parser.add_argument(
        "--training_ratio", type=float, default=0.6, required=False
    )
    parser.add_argument(
        "--validation_ratio", type=float, default=0.01, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, required=False
    )
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.01, required=False
    )
    parser.add_argument(
        "--missing_ratio", type=float, default=0.0, required=False
    )

    config = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    Solver = Solver_LatentRealNVP(data_name=config.data, hidden_dim=config.hidden_dim,
                                  seed=config.seed, z_dim=config.z_dim,
                                  learning_rate=config.learning_rate,
                                  batch_size=config.batch_size, training_ratio=config.training_ratio,
                                  max_epochs=config.max_epochs)

    Solver.pretrain()
    Solver.train()
    Solver.test()
    print("Data {} finished".format(config.data))
