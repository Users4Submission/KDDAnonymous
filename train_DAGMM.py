import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import datetime
import torch.utils.data as data
from torch.autograd import grad
from torch.autograd import Variable
from models.DAGMM import DaGMM
from data_process import RealDataset
import matplotlib.pyplot as plt
from utils import *
import IPython
from tqdm import tqdm


# self, data_name, hidden_dim=128, z_dim=10, svm_output_dim=5, seed=0, learning_rate=1e-3,
#                  start_anomaly_ratio=0.0, decay=0.001, batch_size=128, training_ratio=0.8, validation_ratio=0.1,
#                  weight_decay_for_svm=1e-6, max_epochs=100, co_teaching=0.0
class Solver():
    DEFAULTS = {}

    def __init__(self, data_name, lambda_energy=0.1, lambda_cov_diag=0.005, hidden_dim=128, z_dim=10, seed=0, learning_rate=1e-3, gmm_k=2,
                 batch_size=128, training_ratio=0.8, validation_ratio=0.1, max_epochs=100, missing_ratio=0.0):
        # Data loader
        self.gmm_k = gmm_k
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name + ".npy"
        self.model_save_path = "./trained_model/{}/{}/DAGMM/{}/".format(data_name, missing_ratio, seed)
        self.result_path = "./results/{}/{}/DAGMM/{}/".format(data_name, missing_ratio, seed)
        os.makedirs(self.model_save_path, exist_ok=True)

        self.learning_rate = learning_rate
        self.missing_ratio = missing_ratio
        self.dataset = RealDataset(data_path, missing_ratio=self.missing_ratio)
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_epochs = max_epochs


        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio + validation_ratio))
        # self.n_validation = int(n_sample * validation_ratio)
        self.n_test = n_sample - self.n_train
        print('|data dimension: {}|data noise ratio:{}'.format(self.dataset.__dim__(), self.data_anomaly_ratio))

        training_data, testing_data = data.random_split(dataset=self.dataset,
                                                                         lengths=[
                                                                             self.n_train, self.n_test
                                                                         ])

        self.training_loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
        # self.validation_loader = data.DataLoader(validation_data, batch_size=self.n_validation, shuffle=False)
        self.testing_loader = data.DataLoader(testing_data, batch_size=self.n_test, shuffle=False)
        self.build_model()
        self.print_network()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(input_dim=self.input_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim, n_gmm=self.gmm_k)
        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.learning_rate)
        # Print networks
        self.print_network()

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self):
        num_params = 0
        for p in self.dagmm.parameters():
            num_params += p.numel()
        # print(name)
        # print(model)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.training_loader)

        # # Start with trained model if exists
        # if self.pretrained_model:
        #     start = int(self.pretrained_model.split('_')[0])
        # else:
        #     start = 0

        start = 0
        # Start training
        iter_ctr = 0
        start_time = time.time()
        min_val_loss = 1e+15
        # self.ap_global_train = np.array([0, 0, 0])
        for e in tqdm(range(start, self.max_epochs)):
            for i, (input_data, labels, _) in enumerate(self.training_loader):
                iter_ctr += 1
                start_time = time.time()

                input_data = self.to_var(input_data)

                # training
                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()

                self.dagmm.eval()

            for i, (input_data, labels, _) in enumerate((self.testing_loader)):
                iter_ctr += 1
                start_time = time.time()

                input_data = self.to_var(input_data)

                # validation
                self.dagmm.eval()
                total_loss, sample_energy, recon_error, cov_diag = self.dagmm_step(input_data, validation_flag=True)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()
                # Print out log info
                # if (i + 1) % self.log_step == 0:
                #     elapsed = time.time() - start_time
                #     total_time = ((self.num_epochs * iters_per_epoch) - (e * iters_per_epoch + i)) * elapsed / (
                #                 e * iters_per_epoch + i + 1)
                #     epoch_time = (iters_per_epoch - i) * elapsed / (e * iters_per_epoch + i + 1)
                #
                #     epoch_time = str(datetime.timedelta(seconds=epoch_time))
                #     total_time = str(datetime.timedelta(seconds=total_time))
                #     elapsed = str(datetime.timedelta(seconds=elapsed))
                #
                #     lr_tmp = []
                #     for param_group in self.optimizer.param_groups:
                #         lr_tmp.append(param_group['lr'])
                #     tmplr = np.squeeze(np.array(lr_tmp))
                #
                #     log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                #         elapsed, epoch_time, total_time, e + 1, self.num_epochs, i + 1, iters_per_epoch, tmplr)
                #
                #     for tag, value in loss.items():
                #         log += ", {}: {:.4f}".format(tag, value)
                #
                #     IPython.display.clear_output()
                #     print(log)
                #
                #     if self.use_tensorboard:
                #         for tag, value in loss.items():
                #             self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                #     else:
                #         plt_ctr = 1
                #         if not hasattr(self, "loss_logs"):
                #             self.loss_logs = {}
                #             for loss_key in loss:
                #                 self.loss_logs[loss_key] = [loss[loss_key]]
                #                 plt.subplot(2, 2, plt_ctr)
                #                 plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                #                 plt.legend()
                #                 plt_ctr += 1
                #         else:
                #             for loss_key in loss:
                #                 self.loss_logs[loss_key].append(loss[loss_key])
                #                 plt.subplot(2, 2, plt_ctr)
                #                 plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                #                 plt.legend()
                #                 plt_ctr += 1
                #
                #         plt.show()
                #
                #     print("phi", self.dagmm.phi, "mu", self.dagmm.mu, "cov", self.dagmm.cov)
                # Save model checkpoints

            if loss['total_loss'] < min_val_loss:
                min_val_loss = loss['total_loss']
                torch.save(self.dagmm.state_dict(),
                           os.path.join(self.model_save_path, 'parameter.pth'))

    def dagmm_step(self, input_data, validation_flag=False):
        input_data = input_data.float()
        if not validation_flag:
            self.optimizer.zero_grad()
            self.dagmm.train()

            enc, dec, z, gamma = self.dagmm(input_data)
            if torch.isnan(z.sum()):
                for p in self.dagmm.parameters():
                    print(p)
                print("pause")
            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                        self.lambda_energy,
                                                                                        self.lambda_cov_diag)


            # self.reset_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
            self.optimizer.step()

        else:
            self.dagmm.eval()
            enc, dec, z, gamma = self.dagmm(input_data)

            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma,
                                                                                        self.lambda_energy,
                                                                                        self.lambda_cov_diag)

        return total_loss, sample_energy, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        # self.dagmm.load_stat
        self.dagmm.load_state_dict(torch.load(self.model_save_path + 'parameter.pth'))
        self.dagmm.eval()
        # self.data_loader.dataset.mode = "train"

        # compute the parameter of density estimation by using training and validation set
        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0

        for it, (input_data, labels, _) in enumerate(self.training_loader):

            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
            cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

            N += input_data.size(0)

        train_phi = gamma_sum / N
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        print("N:", N)
        print("phi :\n", train_phi)
        print("mu :\n", train_mu)
        print("cov :\n", train_cov)

        train_energy = []
        train_labels = []
        train_z = []
        for it, (input_data, labels, _) in enumerate(self.training_loader):
            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov,
                                                                size_average=False)

            train_energy.append(sample_energy.data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())


        train_energy = np.concatenate(train_energy, axis=0)
        train_z = np.concatenate(train_z, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)


        test_energy = []
        test_labels = []
        test_z = []
        for it, (input_data, labels, _) in enumerate(self.testing_loader):
            input_data = self.to_var(input_data)
            input_data = input_data.float()
            enc, dec, z, gamma = self.dagmm(input_data)
            sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_energy.append(sample_energy.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy, axis=0)
        test_z = np.concatenate(test_z, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)

        # thresh = np.percentile(combined_energy, 100 - 20)

        thresh = np.percentile(combined_energy, self.data_normaly_ratio * 100)
        # thresh = np.percentile(test_energy, self.data_normaly_ratio * 100)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)


        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(gt, test_energy)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        os.makedirs(self.result_path, exist_ok=True)

        np.save(self.result_path + "result.npy", {
                                            'auc': auc,
                                            'accuracy': accuracy,
                                             'precision': precision,
                                             'recall': recall,
                                             'f1': f_score
                                             })
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision,
                                                                                                    recall, f_score))
        return accuracy, precision, recall, f_score, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument(
        "--seed", type=int, default=0, required=False
    )
    parser.add_argument(
        "--data", type=str, default="optdigits", required=False
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500, required=False
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, required=False
    )
    parser.add_argument(
        "--z_dim", type=int, default=1, required=False
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, required=False
    )
    parser.add_argument(
        "--training_ratio", type=float, default=0.6, required=False
    )
    parser.add_argument(
        "--validation_ratio", type=float, default=0.2, required=False
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, required=False
    )
    parser.add_argument(
        "--start_ratio", type=float, default=0.0, required=False
    )
    parser.add_argument(
        "--data_anomaly_ratio", type=float, default=0.01, required=False
    )
    parser.add_argument(
        "--gmm_k", type=int, default=2, required=False
    )
    parser.add_argument(
        "--missing_ratio", type=float, default=0.0, required=False
    )

    config = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


    DAGMM_Solver = Solver(data_name=config.data, hidden_dim=config.hidden_dim, z_dim=config.z_dim, seed=config.seed,
                          learning_rate=config.learning_rate, gmm_k=config.gmm_k, missing_ratio=config.missing_ratio,
                          batch_size=config.batch_size, training_ratio=config.training_ratio, validation_ratio=config.validation_ratio, max_epochs=config.max_epochs)

    DAGMM_Solver.train()
    DAGMM_Solver.test()
    print("Data {} finished".format(config.data))