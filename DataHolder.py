import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.preprocessing import normalize

class DataHolder:
    def __init__(self, _filename, _number_of_fetures):
        self.rawdata = None
        self.rawdata = self.load_data(_filename, _number_of_fetures)
        self.shuffled_rawdata = self.rawdata.sample(frac=1)

    def load_data(self, _filename, _number_of_features):
        if self.rawdata is None:
            if 'xls' in _filename:
                self.number_of_features = _number_of_features
                return pd.read_excel(_filename, header=None)

    def set_features(self):
        self.features = self.shuffled_rawdata.iloc[1:, 0:self.number_of_features].values

    def get_features(self):
        return self.features

    def set_labels(self):
        self.labels = self.shuffled_rawdata.iloc[1:, -1].values

    def get_labels(self):
        return self.labels

    def encode_labels(self):
        labelencoder_labels = LabelEncoder()
        labels = labelencoder_labels.fit_transform(self.labels)
        labels = np.reshape(labels, (-1, 1))
        onehotencoder = OneHotEncoder(categorical_features=[0])
        labels = onehotencoder.fit_transform(labels).toarray()
        self.encoded_labels  = labels

    def get_encoded_labels(self):
        return self.encoded_labels

    def normalize_features(self):
        self.normalized_features  = normalize(self.features, axis=0, norm='max')

    def get_normalized_features(self):
        return self.normalized_features




