import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def normalize(data):

    for i in range(4):
        norm = np.linalg.norm(data[:, i])
        if norm == 0:
            norm_data[:, i] = data[:, i]
        norm_data = np.array((data[:, i])/norm)
    return norm_data


inputData = pd.read_excel('./IrisDataTrain.xls', header=None)

features = inputData.iloc[1:, 0:4].values
labels = inputData.iloc[1:, 4].values
print(type(features))

norm_features = features/features.max()             # amplitude normalization
plt.figure(1)
plt.scatter(norm_features[:, 0], norm_features[:, 1], color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.figure(2)
plt.scatter(norm_features[:, 2], norm_features[:, 3], color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
#print([norm_features.max(), norm_features.min()])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_labels = LabelEncoder()
labels = labelencoder_labels.fit_transform(labels)
labels = np.reshape(labels,(-1, 1))
onehotencoder = OneHotEncoder(categorical_features = [0])
labels = onehotencoder.fit_transform(labels).toarray()


