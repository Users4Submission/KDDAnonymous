from models.RealNVP import Model_RealNVP
from pyod.models.lof import LOF
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
from K_spectral_norm import spectral_norm
import numpy as np
from utils import *
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.optim import Adam
from data_process import RealDataset, RealGraphDataset, CIFARVGGDataset


class SolverAECIFAR():
    def __init__(self, data_name, hidden_dim=256, seed=0, learning_rate=3e-4, normal_class=0, anomaly_ratio=0.1,
                 batch_size=128, concentrated=0, training_ratio=0.8, SN=1, Trim=1, L=1.5, max_epochs=100):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.L = L
        if concentrated == 1.0:
            full_data_name = 'CIFAR10_Concentrated'
        elif concentrated == 0.0:
            full_data_name = 'CIFAR10'
        self.result_path = "./results/{}_{}_{}/0.0/LOF/{}/".format(
            full_data_name, normal_class, anomaly_ratio, seed
        )
        data_path = "./data/" + data_name + ".npy"
        self.learning_rate = learning_rate
        self.SN = SN
        self.Trim = Trim
        # self.dataset = RealGraphDataset(data_path, missing_ratio=0, radius=2)
        self.dataset = CIFARVGGDataset(data_path, normal_class=normal_class, anomaly_ratio=anomaly_ratio, concentrated=concentrated)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs

        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.batch_size = batch_size
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * training_ratio)
        self.n_test = n_sample - self.n_train
        print('|data dimension: {}|data noise ratio:{}'.format(self.dataset.__dim__(), self.data_anomaly_ratio))

        self.training_data, self.testing_data = data.random_split(dataset=self.dataset,
                                                                         lengths=[
                                                                             self.n_train,
                                                                             self.n_test
                                                                         ])

        self.ae = None
        self.discriminator = None
        self.model=None



    def train(self):
        self.model = LOF()
        self.model.fit(self.training_data.dataset.x)


    def test(self):
        y_test_scores = self.model.decision_function(self.testing_data.dataset.x)
        auc = roc_auc_score(self.testing_data.dataset.y, y_test_scores)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

        print("AUC:{:0.4f}".format(
           auc))

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "accuracy": auc,
                "precision": auc,
                "recall": auc,
                "f1": auc,
                "auc": auc,
            },
        ) # for consistency
        print("result save to {}".format(self.result_path))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument(
        "--seed", type=int, default=5, required=False
    )
    parser.add_argument(
        "--data", type=str, default="vgg19cifar10", required=False
    )
    parser.add_argument(
        "--normal_class", type=int, default=0, required=False
    )
    parser.add_argument(
        "--L", type=float, default=1.5, required=False
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, required=False
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, required=False
    )
    parser.add_argument(
        "--round_D", type=int, default=1, required=False
    )
    parser.add_argument(
        "--z_dim", type=int, default=2, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, required=False
    )
    parser.add_argument(
        "--training_ratio", type=float, default=0.8, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, required=False
    )
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.1, required=False
    )
    parser.add_argument(
        "--missing_ratio", type=float, default=0.0, required=False
    )
    parser.add_argument(
        "--SN", type=int, default=0, required=False
    )
    parser.add_argument(
        "--Trim", type=int, default=1, required=False
    )
    parser.add_argument(
        "--concentrated", type=float, default=1.0, required=False
    )

    config = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    Solver = SolverAECIFAR(data_name=config.data, hidden_dim=config.hidden_dim,
                              seed=config.seed,
                              normal_class=config.normal_class,
                              anomaly_ratio= config.data_anomaly_ratio,
                              learning_rate=config.learning_rate,
                              batch_size=config.batch_size, training_ratio=config.training_ratio,
                              max_epochs=config.max_epochs,
                              L = config.L,
                              concentrated=config.concentrated,
                              SN=config.SN, Trim=config.Trim)

    Solver.train()
    Solver.test()
    print("Data {} finished".format(config.data))
