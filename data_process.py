import os
from torch_geometric.data import Data
import networkx as nx
from six.moves import cPickle as pickle  # for performance
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer,SimpleImputer
# from sklearn.impute import
from copy import deepcopy
import random
import torch as torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import nan_euclidean_distances, euclidean_distances, cosine_distances, pairwise_kernels
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


class MnistDataset(Dataset):
    def __init__(self, path):

        data = np.load(path)
        data = data.item()
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx])),
            torch.from_numpy(np.array(self.y[idx])),
        )


class SyntheticDataset(Dataset):
    """ load synthetic time series data"""

    def __init__(self, path):

        data = np.load(path)
        data = data.item()
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx])),
            torch.from_numpy(np.array(self.y[idx])),
        )


class SyntheticDatasetWithMissing(Dataset):
    """ load synthetic time series data"""

    def __init__(self, path):

        data = np.load(path)
        data = data.item()
        self.x = data["x"]
        self.y = data["y"]
        self.m = data["mask"]

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )


class PytorchGeometricDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super(PytorchGeometricDataset, self).__init__(None, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']



class RealGraphDataset():
    def __init__(self, path, missing_ratio, radius, knn_impute=False):
        scaler = MinMaxScaler()
        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio
        self.x = data["x"]
        self.y = data["y"]

        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        self.m = mask
        if missing_ratio > 0.0:
            self.x[mask == 0] = np.nan
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.x = imputer.fit_transform(self.x)
            self.m = np.ones_like(self.x)
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)
        else:
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)

        similarity = pairwise_kernels(self.x, metric='rbf')
        # plt.bar(sorted(similarity.flatten()))
        threshold = np.percentile(similarity, 99.95)
        self.sparse_graph = similarity > threshold
        # self.sparse_graph = radius_neighbors_graph(self.x, 2, mode='connectivity', metric='minkowski', p=2)
        # self.sparse_graph = kneighbors_graph(self.x, 1, mode='connectivity', metric='minkowski', p=2)
        # self.graph = self.sparse_graph.toarray()
        self.neighbors = {}
        D = nx.DiGraph(self.sparse_graph, with_labels=False, node_size=1)
        # nx.draw_networkx(D)
        # plt.show()
        # x = torch.tensor(data)
        print("number of nodes: {}, number of edges: {}".format(D.number_of_nodes(), D.number_of_edges()))
        edge = D.edges

        edge = [list(ele) for ele in edge]
        edge_index = torch.tensor(edge, dtype=torch.long)

        self.data_torchGeometric = Data(x=torch.tensor(self.x), edge_index=edge_index)


    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class CIFARVGGDataset(Dataset):
    def __init__(self, path, normal_class, anomaly_ratio, concentrated):
        scaler = MinMaxScaler()
        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.x = data["x"]
        self.y = data["y"]
        self.x = np.concatenate(self.x, axis=0)
        self.x = self.x.squeeze()

        pca = PCA(n_components=128)
        self.x = pca.fit_transform(self.x)

        self.y = np.concatenate(self.y)
        self.y = self.y.squeeze()

        if concentrated == 0.0:
            normal_index = np.where(self.y == normal_class)[0]
            anomaly_index = np.where(self.y != normal_class)[0]

            y = np.zeros_like(self.y)
            y[anomaly_index] = 1

            normal_x = self.x[normal_index, :]
            normal_y = y[normal_index]

            anomaly_x = self.x[anomaly_index, :]
            anomaly_y = y[anomaly_index]

            n_normal = normal_x.shape[0]
            n_anomaly = int(n_normal * anomaly_ratio)

            full_index = np.arange(anomaly_y.sum())
            random.shuffle(full_index)
            selected_anomaly = full_index[:n_anomaly]
            anomaly_x_subset = anomaly_x[selected_anomaly, :]
            anomaly_y_subset = anomaly_y[selected_anomaly]

            self.x = np.concatenate((normal_x, anomaly_x_subset), axis=0)
            self.y = np.concatenate((normal_y, anomaly_y_subset))
        elif concentrated == 1.0:
            normal_index = np.where(self.y == normal_class)[0]
            anomaly_index = np.where(self.y == (normal_class+1)%10)[0]

            y = np.zeros_like(self.y)
            y[anomaly_index] = 1

            normal_x = self.x[normal_index, :]
            normal_y = y[normal_index]

            anomaly_x = self.x[anomaly_index, :]
            anomaly_y = y[anomaly_index]

            n_normal = normal_x.shape[0]
            n_anomaly = int(n_normal * anomaly_ratio)

            full_index = np.arange(anomaly_y.sum())
            random.shuffle(full_index)
            selected_anomaly = full_index[:n_anomaly]
            anomaly_x_subset = anomaly_x[selected_anomaly, :]
            anomaly_y_subset = anomaly_y[selected_anomaly]

            self.x = np.concatenate((normal_x, anomaly_x_subset), axis=0)
            self.y = np.concatenate((normal_y, anomaly_y_subset))



        n, d = self.x.shape
        mask = np.random.rand(n, d)
        self.m = mask

        scaler.fit(self.x)
        self.x = scaler.transform(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class LimnoTemporalDataset(Dataset):
    def __init__(self, x):
        scaler = MinMaxScaler()
        self.x = x.to_numpy()
        n, d = self.x.shape
        # self.x[:, :6] = scaler.fit_transform(self.x[:, :6])

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

class RealDataset(Dataset):
    def __init__(self, path, missing_ratio, knn_impute=False):
        scaler = MinMaxScaler()

        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio
        self.x = data["x"]
        self.y = data["y"]

        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        self.m = mask
        if missing_ratio > 0.0:
            self.x[mask == 0] = np.nan
            # imputer = KNNImputer(n_neighbors=2)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.x = imputer.fit_transform(self.x)
            self.m = np.ones_like(self.x)

            scaler.fit(self.x)
            self.x = scaler.transform(self.x)
        else:
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)


        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]

def normalize(x):
    return (x - 128.0) / 128


class CIFARPretrainDataset(Dataset):
    """ load synthetic time series data"""

    def __init__(self, x):
        # self.x = normalize(np.transpose(x, (0, 3, 1, 2)))
        # self.x = (np.transpose(x, (0, 3, 1, 2)))
        self.x = x
        # standardize images
        # for i in range(self.x.shape[0]):
        #     x[i, :, :, :] = x[i, :, :, :] - x[i, :, :, :].mean()
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
            return self.x.shape[1:]

    def __getitem__(self, idx):
        # print(self.transform(self.x[idx, :, :, :]))
        return (
            self.transform(self.x[idx, :, :, :]),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class CIFARDataset(Dataset):
    """ load synthetic time series data"""

    def __init__(self, x, y, normal_class):
        # self.x = normalize(np.transpose(x, (0, 3, 1, 2)))
        # self.x = (np.transpose(x, (0, 3, 1, 2)))
        self.x = x
        # standardize images
        # for i in range(self.x.shape[0]):
        #     x[i, :, :, :] = x[i, :, :, :] - x[i, :, :, :].mean()
        self.y = (y != normal_class).astype(float)
        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ])
        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
            return self.x.shape[1:]

    def __getitem__(self, idx):
        # print(self.transform(self.x[idx, :, :, :]))
        return (
            self.transform(self.x[idx, :, :, :]),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.y[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class OracleDataset(Dataset):
    """ load synthetic time series data"""

    def __init__(self, path, missing_ratio, knn_impute=False):
        scaler = MinMaxScaler()

        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio

        self.x = data["x"]
        self.y = data["y"]

        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        self.m = mask
        if missing_ratio > 0.0:
            self.x[mask == 0] = np.nan
            # imputer = KNNImputer(n_neighbors=2)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.x = imputer.fit_transform(self.x)
            self.m = np.ones_like(self.x)
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)
        else:
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)


        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class RealDataset_KNN(Dataset):
    """ load synthetic time series data"""

    def __init__(self, path, missing_ratio, knn_impute=False, n_neighbor=5):
        scaler = MinMaxScaler()
        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio

        self.x = data["x"]
        self.y = data["y"]

        """ the argsort of numpy seems have bugs"""
        if missing_ratio == 0.0:
            pdist_x = cosine_distances(self.x, self.x)
        else:
            pdist_x = nan_euclidean_distances(self.x, self.x)
        graph_knn = np.zeros_like(pdist_x)
        for i in range(self.x.shape[0]):
            sorted_row = np.sort(pdist_x[i, :])
            sort_index = np.argsort(pdist_x[i, :])

            '''deal with when there are multiple point is the same to guarantee the diag is 0'''
            if sort_index[0] != i:
                j = np.where(sort_index == i)
                sort_index[j] = sort_index[0]
                sort_index[0] = i

            # if i==266:
            #     print(sorted_row)
            selected_index = sort_index[:1+n_neighbor]
            graph_knn[i, selected_index] = 1
            # print(sort_index[:20])
            thresh = sorted_row[n_neighbor + 1]
            # if thresh == sorted_row[n_neighbor + 2]:
            #     print(sorted_row)
            pdist_x[i, :] = (pdist_x[i, :] < thresh).astype(float)
            # pdist_x[i,:] = 0
            # pdist_x[i, sort_index[1:1+n_neighbor]]=1
        # graph_knn = pdist_x - np.eye(self.x.shape[0])
        graph_knn = graph_knn - np.eye(self.x.shape[0])
        np.testing.assert_equal(
            np.sum(graph_knn, axis=1), n_neighbor * np.ones(self.x.shape[0])
        )


        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        self.m = mask
        if missing_ratio > 0.0:

            self.x[mask == 0] = np.nan
            # imputer = KNNImputer(n_neighbors=2)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.x = imputer.fit_transform(self.x)
            self.m = np.ones_like(self.x)
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)

        '''get nn'''
        x_nn = []
        for i in range(self.x.shape[0]):
            index = np.where(graph_knn[i, :] == 1)
            # if index[0].shape[0] != 5:
            #     print(i)
            x_i = self.x[index, :]

            x_nn.append(x_i.squeeze())
        self.x_nn = np.stack(x_nn)
        # nn_mat = np.concatenate(nn_mat)


        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.x_nn[idx, :, :])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]


class RealDataset_Limno(Dataset):
    """ load synthetic time series data"""

    def __init__(self, path, missing_ratio, knn_impute=False, n_neighbor=5):
        scaler = MinMaxScaler()
        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio

        self.x = data["x"]
        self.y = data["y"]
        self.lagosid = data['lagosid']

        n = self.x.shape[0]
        # random_index = np.random.choice(n, 50000, replace=False)
        # random_index = np.array([0:50000])
        self.x = self.x[:50000, :]
        self.y = self.y[:50000]
        self.lagosid = self.lagosid[:50000]
        """ the argsort of numpy seems have bugs"""
        pdist_x = cosine_distances(self.x, self.x)
        graph_knn = np.zeros_like(pdist_x)
        for i in range(self.x.shape[0]):
            sorted_row = np.sort(pdist_x[i, :])
            sort_index = np.argsort(pdist_x[i, :])

            '''deal with when there are multiple point is the same to guarantee the diag is 0'''
            if sort_index[0] != i:
                j = np.where(sort_index==i)
                sort_index[j] = sort_index[0]
                sort_index[0] = i

            # if i==266:
            #     print(sorted_row)
            selected_index = sort_index[:1+n_neighbor]
            graph_knn[i, selected_index] = 1
            # print(sort_index[:20])
            thresh = sorted_row[n_neighbor + 1]
            # if thresh == sorted_row[n_neighbor + 2]:
            #     print(sorted_row)
            pdist_x[i, :] = (pdist_x[i, :] < thresh).astype(float)
            # pdist_x[i,:] = 0
            # pdist_x[i, sort_index[1:1+n_neighbor]]=1
        # graph_knn = pdist_x - np.eye(self.x.shape[0])
        graph_knn = graph_knn - np.eye(self.x.shape[0])
        np.testing.assert_equal(
            np.sum(graph_knn, axis=1), n_neighbor * np.ones(self.x.shape[0])
        )

        '''get nn'''
        x_nn = []
        for i in range(self.x.shape[0]):
            index = np.where(graph_knn[i, :] == 1)
            # if index[0].shape[0] != 5:
            #     print(i)
            x_i = self.x[index, :]

            x_nn.append(x_i.squeeze())
        self.x_nn = np.stack(x_nn)
        # nn_mat = np.concatenate(nn_mat)
        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        self.m = mask
        if knn_impute:
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)
            self.x[mask == 0] = np.nan
            imputer = KNNImputer(n_neighbors=2)
            self.x = imputer.fit_transform(self.x)
            self.m = np.ones_like(self.x)

        # print(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.x_nn[idx, :, :])),
            torch.from_numpy(np.array(self.y[idx])),
            torch.from_numpy(np.array(self.m[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]
