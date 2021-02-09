import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from scipy.io import loadmat
import argparse
import mat73

def mat2np(data_name):
    path = './data/{}.mat'.format(data_name)
    if data_name == 'http':
        data = mat73.loadmat(path)
    else:
        data = loadmat(path)
    x = data['X']
    y = data['y']
    if data_name == 'optdigits' or data_name == 'mnist' or data_name == 'arrhythmia':
        range_x = (np.amax(x, axis=0)-np.amin(x, axis=0))
        range_x[range_x == 0] = 1
        x = (x - np.amin(x, axis=0))/range_x
    else:
        x = zscore(x)

    if data_name == 'locus':
        lagosid = data['Lagosid']
        np.save("./data/{}.npy".format(data_name), {'x': x, 'y': y, 'lagosid': lagosid})
    else:
        np.save("./data/{}.npy".format(data_name), {'x': x, 'y': y})
    print("{} data has been processed".format(data_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="LoadData")
    parser.add_argument("--filename", type=str, default="kdd_cup.npz", required=False)
    config = parser.parse_args()
    filename = config.filename
    # mat2np(filename)
    # filename = 'sensor'
    # filename = 'online_shoppers_intention'
    # filename = 'kdd_cup.npz'
    # data = np.loadtxt("./data/Sensorless_drive_diagnosis.txt")
    if filename == 'mulcross':
        data = pd.read_csv('./data/phpGGVhl9.csv')
        data = data.to_numpy()
        x = data[:, :4].astype(float)
        y = (data[:, -1]!="'Normal'").astype(int)
        np.save("./data/mulcross.npy", {'x':x, 'y':y})
    if filename in ['cardio',
                     'arrhythmia',
                    'glass',
                    'ionosphere',
                    'vertebral',
                    'vowels',
                    'pendigits',
                    'annthyroid',
                    "http",
                    'letter',
                    'satimage-2',
                    'satellite',
                    'thyroid',
                    'musk',
                    'shuttle',
                    'optdigits',
                    'locus',
                    'speech',
                    'mnist',
                    'mammography',
                    'pima',
                    'breastw',
                    'wbc',
                    "lympho",
                    "ecoli",
                    "glass"
                    ]:
        mat2np(filename)

    if filename == 'sensor':
        data = np.loadtxt("./data/Sensorless_drive_diagnosis.txt")
        x = data[:, :-1]
        y = data[:, -1]
        y = y==11
        y = y.astype(float)
        x = zscore(x)
        np.save("./data/sensor.npy", {'x': x, 'y': y})

    if filename == 'kdd_cup.npz':
        url_base = "http://kdd.ics.uci.edu/databases/kddcup99"

        # KDDCup 10% Data
        url_data = f"{url_base}/kddcup.data_10_percent.gz"
        # info data (column names, col types)
        url_info = f"{url_base}/kddcup.names"
        df_info = pd.read_csv(url_info, sep=":", skiprows=1, index_col=False, names=["colname", "type"])
        colnames = df_info.colname.values
        coltypes = np.where(df_info["type"].str.contains("continuous"), "float", "str")
        colnames = np.append(colnames, ["status"])
        coltypes = np.append(coltypes, ["str"])

        # Import data
        df = pd.read_csv(url_data, names=colnames, index_col=False,
                         dtype=dict(zip(colnames, coltypes)))
        X = pd.get_dummies(df.iloc[:, :-1]).values

        Scaler = StandardScaler()
        X = Scaler.fit_transform(X)
        # Create Traget Flag
        # Anomaly data when status is normal, Otherwise, Not anomaly.
        y = np.where(df.status == "normal.", 1, 0)
        # data = np.load("./data/{}".format(filename))
        # y = data["kdd"][:, -1]
        # x = data["kdd"][:, :-1]
        np.save("./data/kddcup.npy", {'x':X, 'y':y})

    if filename == 'online_shoppers_intention':
        data = pd.read_csv("./data/{}.csv".format(filename), header=0, sep=',', names=['Administrative',
                                                                         'Administrative_Duration',
                                                                         'Informational	Informational_Duration',
                                                                         'ProductRelated',
                                                                         'ProductRelated_Duration',
                                                                         'BounceRates',
                                                                         'ExitRates',
                                                                         'PageValues',
                                                                         'SpecialDay',
                                                                         'Month',
                                                                         'OperatingSystems',
                                                                         'Browser',
                                                                         'Region',
                                                                         'TrafficType',
                                                                         'VisitorType',
                                                                         'Weekend',
                                                                         'Revenue'])

        one_hot_month = pd.get_dummies(data["Month"], drop_first=True)
        one_hot_operatingsystems = pd.get_dummies(data["OperatingSystems"], drop_first=True)
        one_hot_browser = pd.get_dummies(data["Browser"], drop_first=True)
        one_hot_region = pd.get_dummies(data["Region"], drop_first=True)
        one_hot_traffictype = pd.get_dummies(data["TrafficType"], drop_first=True)
        one_hot_vistortype = pd.get_dummies(data["VisitorType"], drop_first=True)
        one_hot_weekend = pd.get_dummies(data["Weekend"], drop_first=True)
        one_hot_revenue = pd.get_dummies(data["Revenue"], drop_first=True)

        data = data.drop("Month", axis=1)
        data = data.drop("OperatingSystems", axis=1)
        data = data.drop("Browser", axis=1)
        data = data.drop("Region", axis=1)
        data = data.drop("TrafficType", axis=1)
        data = data.drop("VisitorType", axis=1)
        data = data.drop("Weekend", axis=1)
        data = data.drop("Revenue", axis=1)


        data = pd.concat([data,
                          one_hot_month,
                          one_hot_operatingsystems,
                          one_hot_browser,
                          one_hot_region,
                          one_hot_traffictype,
                          one_hot_vistortype,
                          one_hot_weekend,
                          one_hot_revenue], axis=1)

        data = pd.DataFrame(data)
        x = data.to_numpy()

        y = x[:, -1]
        x = x[:, :-1]

        np.save("./data/{}.npy".format(filename), {'x':x, 'y':y})
    print("data processing finished")