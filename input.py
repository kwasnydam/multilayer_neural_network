import numpy as np

import pandas as pd


def normalize(data):

    for i in range(4):
        norm = np.linalg.norm(data[:, i])
        if norm == 0:
            norm_data[:,i] = data[:, i]
        norm_data = np.array((data[:, i])/norm)
    return norm_data

inputData = pd.read_excel('./IrisDataTrain.xls', header=None)

features = inputData.iloc[1:, 0:4]
labels = inputData.iloc[1:, 4]
print(type(features))

features = features.values
labels = labels.values
print([features])



norm_features = normalize(features)


print([norm_features])



