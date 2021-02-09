import torch as torch
import torch.nn as nn
import os

# from torch.utils import data
import torch.utils.data as data
import time
import numpy as np
import copy
from tqdm import tqdm

from sklearn.impute import KNNImputer
import argparse
from loss import VAE_LOSS_SCORE, VAE_LOSS, VAE_Outlier_SCORE
from models.VAE_Siamese import AE
from data_process import SyntheticDataset, RealDataset


class Solver_AE_Coteaching:
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
        coteaching=1.0,
        missing_ratio=0.0,
        knn_impute=False,
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        use_cuda = torch.cuda.is_available()
        self.knn_impute = knn_impute
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name + ".npy"
        self.missing_ratio = missing_ratio
        self.model_save_path = "./trained_model/{}/{}/Coteaching_VAE/{}/".format(
            data_name, missing_ratio, seed
        )
        self.result_path = "./results/{}/{}/Coteaching_VAE/{}/".format(
            data_name, missing_ratio, seed
        )
        os.makedirs(self.model_save_path, exist_ok=True)
        self.learning_rate = learning_rate
        self.dataset = RealDataset(
            data_path, missing_ratio=self.missing_ratio, knn_impute=knn_impute
        )
        self.data_name = data_name
        self.seed = seed
        self.start_ratio = start_ratio

        self.decay_ratio = decay_ratio
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.start_ratio = start_ratio
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
        self.ae.float()

    def print_network(self):
        num_params = 0
        for p in self.ae.parameters():
            num_params += p.numel()
        print("The number of parameters: {}".format(num_params))

    def train(self):
        if self.data_name == 'optdigits':
            loss_type = 'BCE'
        else:
            loss_type = 'MSE'
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.learning_rate)
        mse_loss = torch.nn.MSELoss()
        vae_loss = VAE_LOSS()
        vae_score = VAE_LOSS_SCORE()

        min_val_error = 1e10
        for epoch in tqdm(range(self.max_epochs)):  # train 3 time classifier
            for i, (x, y, m) in enumerate(self.training_loader):
                x = x.to(self.device)
                x = x.float()
                m = m.to(self.device).float()
                n = x.shape[0]
                n_selected = n

                if config.coteaching == 0.0:
                    n_selected = n
                if i == 0:
                    current_ratio = "{}/{}".format(n_selected, n)
                optimizer.zero_grad()
                with torch.no_grad():
                    self.ae.eval()

                    z1, z2, xhat1, xhat2, mu1, mu2, logvar1, logvar2 = self.ae(x.float(), x.float())
                    error1 = vae_score(xhat1, x, mu1, logvar1)
                    error2 = vae_score(xhat2, x, mu2, logvar2)

                    _, index1 = torch.sort(error1)
                    _, index2 = torch.sort(error2)

                    index1 = index1[:n_selected]
                    index2 = index2[:n_selected]

                    x1 = x[index2, :]
                    x2 = x[index1, :]
                    m1 = m[index2, :]
                    m2 = m[index1, :]

                self.ae.train()
                z1, z2, xhat1, xhat2, mu1, logvar1, mu2, logvar2 = self.ae(x1.float(), x2.float())
                loss = vae_loss(xhat1, x, mu1, logvar1, loss_type) + vae_loss(xhat2, x, mu2, logvar2, loss_type)
                loss.backward()
                optimizer.step()
            #
            # if self.start_ratio < self.data_anomaly_ratio:
            #     self.start_ratio = min(
            #         self.data_anomaly_ratio, self.start_ratio + self.decay_ratio
            #     )
            # if self.start_ratio > self.data_anomaly_ratio:
            #     self.start_ratio = max(
            #         self.data_anomaly_ratio, self.start_ratio - self.decay_ratio
            #     )  # 0.0005 for 0.1 anomaly, 0.0001 for 0.001 anomaly

            # with torch.no_grad():
            #     self.ae.eval()
            #     for i, (x, y, m) in enumerate(self.testing_loader):
            #         x = x.to(self.device)
            #         m = m.to(self.device).float()
            #         # y = y.to(device)
            #         x = x.float()
            #         _, _, xhat1, xhat2, mu1, mu2, logvar1, logvar2 = self.ae(x, x, m, m)
            #         error1 = vae_score(xhat1, x, mu1, logvar1, loss_type)
            #         error2 = vae_score(xhat2, x, mu2, logvar2, loss_type)
            #
            #         n_non_missing = m.sum(dim=1)
            #         error1 = error1 / n_non_missing
            #         error2 = error2 / n_non_missing
            #
            #         n_val = x.shape[0]
            #         n_selected = int(n_val * (1 - self.data_anomaly_ratio))
            #         if self.coteaching == 0.0:
            #             n_selected = n
            #         _, index1 = torch.sort(error1)
            #         _, index2 = torch.sort(error2)
            #         index1 = index1[:n_selected]
            #         index2 = index2[:n_selected]
            #
            #         x1 = x[index2, :]
            #         x2 = x[index1, :]
            #         m1 = m[index2, :]
            #         m2 = m[index1, :]
            #         z1, z2, xhat1, xhat2, mu1, mu2, logvar1, logvar2 = self.ae(x1, x2, m1, m2)
            #         val_loss = vae_loss(xhat1, x1, mu1, logvar1, loss_type) + vae_loss(xhat2, x2, mu2, logvar2, loss_type)
            #
            #         if val_loss < min_val_error:
            #             min_val_error = val_loss
            #             torch.save(
            #                 self.ae.state_dict(),
            #                 os.path.join(self.model_save_path, "parameter.pth"),
            #             )

    def test(self):
        print("======================TEST MODE======================")
        # self.dagmm.load_stat
        self.ae.load_state_dict(torch.load(self.model_save_path + "parameter.pth"))
        self.ae.eval()
        vae_loss = VAE_LOSS()
        vae_score = VAE_Outlier_SCORE()

        if self.data_name == 'optdigits':
            loss_type = 'BCE'
        else:
            loss_type = 'MSE'

        for _, (x, y, m) in enumerate(self.testing_loader):
            y = y.data.cpu().numpy()
            x = x.to(self.device).float()
            m = m.to(self.device).float()
            _, _, xhat1, xhat2, mu1, mu2, logvar1, logvar2 = self.ae(x.float(), x.float(), m, m)
            error1 = vae_score(xhat1, x, mu1, logvar1, loss_type)
            error2 = vae_score(xhat2, x, mu2, logvar2, loss_type)
            n_non_missing = m.sum(dim=1)
            error = (
                error1 / n_non_missing + error2 / n_non_missing
            )

        error = error.data.cpu().numpy()
        thresh = np.percentile(error, self.data_normaly_ratio * 100)
        print("Threshold :", thresh)

        pred = (error > thresh).astype(int)
        gt = y.astype(int)

        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )

        auc = roc_auc_score(gt, error)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
                accuracy, precision, recall, f_score, auc
            )
        )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument("--algorithm", type=str, default="AutoEncoder", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--decay", type=float, default=0.001, required=False)
    parser.add_argument("--data", type=str, default="letter", required=False)
    parser.add_argument("--max_epochs", type=int, default=200, required=False)
    parser.add_argument("--knn_impute", type=bool, default=False, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.599, required=False)
    parser.add_argument("--validation_ratio", type=float, default=0.001, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-4, required=False)
    parser.add_argument("--start_ratio", type=float, default=0.0, required=False)
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.01, required=False
    )
    parser.add_argument("--z_dim", type=int, default=10, required=False)
    parser.add_argument("--coteaching", type=float, default=1.0, required=False)
    parser.add_argument("--missing_ratio", type=float, default=0.0, required=False)
    config = parser.parse_args()

    """
    read data
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True

    Solver = Solver_AE_Coteaching(
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
        knn_impute=config.knn_impute,
    )

    Solver.train()
    Solver.test()
    print("Data {} finished".format(config.data))
