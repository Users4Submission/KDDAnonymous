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

from models.AE_Coteaching import AE
from data_process import SyntheticDataset, RealDataset


class Solver_AE:
    def __init__(
        self,
        data_name,
        start_ratio=0.0,
        decay_ratio=0.01,
        hidden_dim=128,
        z_dim=10,
        seed=0,
        learning_rate=1e-3,
        batch_size=128,
        training_ratio=0.8,
        validation_ratio=0.1,
        max_epochs=100,
        coteaching=0.0,
        knn_impute=False,
        missing_ratio=0.0
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name + ".npy"
        self.model_save_path = "./trained_model/{}/{}/AE/{}/".format(data_name, missing_ratio, seed)
        self.result_path = "./results/{}/{}/AE/{}/".format(data_name, missing_ratio, seed)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.learning_rate = learning_rate
        self.missing_ratio = missing_ratio
        self.dataset = RealDataset(data_path, missing_ratio=self.missing_ratio, knn_impute=knn_impute)
        self.seed = seed
        self.start_ratio = start_ratio
        self.decay_ratio = decay_ratio
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.coteaching = coteaching

        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio + validation_ratio))
        # self.n_validation = int(n_sample * validation_ratio)
        self.n_test = n_sample - self.n_train
        print(
            "|data dimension: {}|data noise ratio:{}".format(
                self.dataset.__dim__(), self.data_anomaly_ratio
            )
        )

        self.decay_ratio = abs(self.start_ratio - (1 - self.data_anomaly_ratio)) / (
            self.max_epochs / 2
        )
        training_data, testing_data = data.random_split(
            dataset=self.dataset, lengths=[self.n_train, self.n_test]
        )

        self.training_loader = data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True
        )

        self.testing_loader = data.DataLoader(
            testing_data, batch_size=self.n_test, shuffle=False
        )
        self.ae = None
        self.discriminator = None
        self.build_model()
        self.print_network()

    def build_model(self):
        self.ae = AE(
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
        mse_loss = torch.nn.MSELoss()
        if self.data_name == 'optdigits':
            mse_loss = torch.nn.BCELoss()
        min_val_error = 1e10
        for epoch in tqdm(range(self.max_epochs)):  # train 3 time classifier
            for i, (x, y, m) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                m = m.to(self.device).float()

                # x_missing = x * m + (1-m) * -10
                n = x.shape[0]
                optimizer.zero_grad()
                self.ae.train()
                z1, z2, xhat1, xhat2 = self.ae(x.float(), x.float(), m, m)

                loss = mse_loss(xhat1, x) + mse_loss(xhat2, x)
                print(loss.item())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.ae.eval()
                for i, (x, y, m) in enumerate(self.testing_loader):
                    x = x.to(self.device)
                    m = m.to(self.device).float()
                    # y = y.to(device)
                    x = x.float()
                    _, _, xhat1, xhat2 = self.ae(x, x, m, m)
                    val_loss = mse_loss(xhat1, x) + mse_loss(xhat2, x)
                    if val_loss < min_val_error:
                        # print(epoch)
                        min_val_error = val_loss
                        torch.save(
                            self.ae.state_dict(),
                            os.path.join(self.model_save_path, "parameter.pth"),
                        )
            # scheduler.step()

    def test(self):
        print("======================TEST MODE======================")
        self.ae.load_state_dict(torch.load(self.model_save_path + "parameter.pth"))
        self.ae.eval()
        loss = torch.nn.MSELoss(reduction='none')
        if self.data_name == 'optdigits':
            loss = torch.nn.BCELoss(reduction='none')

        for _, (x, y, m) in enumerate(self.testing_loader):
            y = y.data.cpu().numpy()
            x = x.to(self.device).float()
            m = m.to(self.device).float()

            _, _, xhat1, xhat2 = self.ae(x.float(), x.float(), m, m)
            error = loss(xhat1, x) + loss(xhat2, x)
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
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy, precision, recall, f_score
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
        return accuracy, precision, recall, f_score, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument("--algorithm", type=str, default="AutoEncoder", required=False)
    parser.add_argument("--seed", type=int, default=5, required=False)
    parser.add_argument("--decay", type=float, default=0.001, required=False)
    parser.add_argument("--data", type=str, default="torchcifar10_0.9_9", required=False)
    parser.add_argument("--max_epochs", type=int, default=50, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.599, required=False)
    parser.add_argument("--validation_ratio", type=float, default=0.001, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-4, required=False)
    parser.add_argument("--start_ratio", type=float, default=0.0, required=False)
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.01, required=False
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
        data_name=config.data,
        hidden_dim=config.hidden_dim,
        z_dim=config.z_dim,
        seed=config.seed,
        start_ratio=config.start_ratio,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        decay_ratio=config.decay,
        training_ratio=config.training_ratio,
        validation_ratio=config.validation_ratio,
        max_epochs=config.max_epochs,
        missing_ratio=config.missing_ratio,
        knn_impute = config.knn_impute
    )

    Solver.train()
    Solver.test()
    print("Data {} finished".format(config.data))
