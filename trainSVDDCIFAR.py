import torch as torch
import torch.nn as nn
import os

# from torch.utils import data
import torch.utils.data as data
import time
import numpy as np
import copy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import argparse

from models.SVDD import SVDD, SVMLoss
from data_process import SyntheticDataset, RealDataset, CIFARVGGDataset


class Solver_AE:
    def __init__(
        self,
        hidden_dim=128,
        z_dim=10,
        seed=0,
        concentrated=0,
        normal_class=0,
        learning_rate=3e-4,
        batch_size=128,
        training_ratio=0.8,
        max_epochs=100,
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        use_cuda = torch.cuda.is_available()
        data_name = 'vgg19cifar10'
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if concentrated == 1.0:
            full_data_name = 'CIFAR10_Concentrated'
        elif concentrated == 0.0:
            full_data_name = 'CIFAR10'
        anomaly_ratio = 0.1
        self.result_path = "./results/{}_{}_{}/0.0/SVDD/{}/".format(
            full_data_name, anomaly_ratio, normal_class, seed
        )
        self.data_anomaly_ratio = 0.1
        data_path = "./data/" + data_name + ".npy"
        self.learning_rate = learning_rate
        # self.dataset = RealGraphDataset(data_path, missing_ratio=0, radius=2)
        self.dataset = CIFARVGGDataset(data_path, normal_class=normal_class, anomaly_ratio=anomaly_ratio, concentrated=concentrated)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.input_dim = 128
        self.data_normaly_ratio = 1 - anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio))
        self.n_test = n_sample - self.n_train
        print('|data dimension: {}|data noise ratio:{}'.format(self.dataset.__dim__(), self.data_anomaly_ratio))

        training_data, testing_data = data.random_split(dataset=self.dataset,
                                                                         lengths=[
                                                                             self.n_train, self.n_test
                                                                         ])

        self.training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        self.testing_loader = data.DataLoader(testing_data, batch_size=batch_size, shuffle=False)

        self.ae = None
        self.discriminator = None
        self.build_model()
        self.print_network()

    def build_model(self):
        self.ae = SVDD(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim
        )
        self.ae = self.ae.to(self.device)

    def print_network(self):
        num_params = 0
        for p in self.ae.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def train(self):
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        """
        pretrain autoencoder
        """
        mse_loss = torch.nn.MSELoss()
        for epoch in tqdm(range(50)):  # train 3 time classifier
            for i, (x, y, m) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                m = m.to(self.device).float()

                # x_missing = x * m + (1-m) * -10
                n = x.shape[0]
                optimizer.zero_grad()
                self.ae.train()
                z1, xhat1, _ = self.ae(x.float(), m)

                loss = mse_loss(xhat1, x)
                loss.backward()
                optimizer.step()
            # scheduler.step()
        # svm
        #init c
        svm_loss = SVMLoss()
        z = []
        with torch.no_grad():
            self.ae.eval()
            for i, (x, y, m) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                m = m.to(self.device).float()

                z1, _, _ = self.ae(x.float(), m.float())
                z.append(z1)
                # x_intersect = x[index_intersect, :]
            z = torch.cat(z).mean(dim=0)
            center = self.ae.init_c(z)

        self.ae.train()
        for epoch in range(self.max_epochs):
            for i, (x, y, m) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                m = m.to(self.device).float()
                n = x.shape[0]
                optimizer.zero_grad()
                z1, _, _ = self.ae(x.float(), m)
                loss = svm_loss(z1, center)
                loss.backward()
                optimizer.step()

    def test(self):
        print("======================TEST MODE======================")
        self.ae.eval()
        loss = SVMLoss()

        for _, (x, y, m) in enumerate(self.testing_loader):
            y = y.data.cpu().numpy()
            x = x.to(self.device).float()
            m = m.to(self.device).float()

            z1, _, _ = self.ae(x.float(), m)
            error = ((z1 - self.ae.c1)**2)
            error = error.sum(dim=1)
        error = error.data.cpu().numpy()
        thresh = np.percentile(error, self.data_normaly_ratio * 100)
        print("Threshold :", thresh)

        pred = (error > thresh).astype(int)
        gt = y.astype(int)

        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score
        )
        gt = gt.squeeze()
        auc = roc_auc_score(gt, error)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, auc: {:0.4f}".format(
                accuracy, precision, recall, f_score, auc
            )
        )

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
            },
        )
        print("result save to {}".format(self.result_path))
        return accuracy, precision, recall, f_score, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument("--algorithm", type=str, default="AutoEncoder", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--max_epochs", type=int, default=200, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.8, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-4, required=False)
    parser.add_argument("--start_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--normal_class", type=int, default=0, required=False)
    parser.add_argument("--concentrated", type=int, default=0, required=False)
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.1, required=False
    )
    parser.add_argument("--z_dim", type=int, default=10, required=False)
    parser.add_argument("--missing_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--knn_impute", type=bool, default=True, required=False)

    config = parser.parse_args()

    """
    read data
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True

    Solver = Solver_AE(
        hidden_dim=config.hidden_dim,
        z_dim=config.z_dim,
        seed=config.seed,
        concentrated=config.concentrated,
        normal_class=config.normal_class,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        training_ratio=config.training_ratio,
        max_epochs=config.max_epochs,
    )

    Solver.train()
    Solver.test()

