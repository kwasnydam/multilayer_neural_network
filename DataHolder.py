import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.preprocessing import normalize

class DataHolder:
    def __init__(self, _filename, _number_of_fetures):
        self.rawdata = None
        try:
            self.rawdata = self.load_data(_filename, _number_of_fetures)
            if self.rawdata is None:
                raise Exception
            else:
                self.shuffled_rawdata = self.rawdata.sample(frac=1)
                print('Data loaded succesfully!')
                self.data_loaded_succesfully = True
                self.set_features()
                self.set_labels()
                self.are_labels_encoded = False
                self.are_features_normalized = False
        except:
            print('Failed to load the data')

    def load_data(self, _filename, _number_of_features):
        try:
            if 'xls' in _filename:
                self.number_of_features = _number_of_features
                try:
                    data = pd.read_excel(_filename, header=None)
                except FileNotFoundError as e:
                    print('Problem Loading File')
                else:
                    return data
            elif 'csv' in _filename:
                self.number_of_features = _number_of_features
                try:
                    data = pd.read_csv(_filename, header=None)
                except FileNotFoundError as e:
                    print('Problem Loading File')
                else:
                    return data
            else:
                raise Exception('Wrong filetype!')
        except Exception as e:
            print(e)
        finally:
            print("we are still running bois")

    def set_features(self):
        self.features = self.shuffled_rawdata.iloc[1:, 0:self.number_of_features].values
        pass

    def get_features(self):
        if not self.are_features_normalized:
            return self.features
        else:
            return self.normalized_features


    def set_labels(self):
        self.labels = self.shuffled_rawdata.iloc[1:, -1].values  # Takes last column of input data as labels
        #self.labels = np.reshape(labels, (-1, 1))           # Reshapes it to be the coulmn vector
        pass

    def get_labels(self):
        if not self.are_labels_encoded:
            return self.labels
        else:
            return self.encoded_labels

    def encode_labels(self):
        labelencoder_labels = LabelEncoder()
        labels = labelencoder_labels.fit_transform(self.labels)
        labels = np.reshape(labels, (-1, 1))
        onehotencoder = OneHotEncoder(categorical_features=[0])
        labels = onehotencoder.fit_transform(labels).toarray()
        self.encoded_labels  = labels
        self.are_labels_encoded = True

    def get_encoded_labels(self):
        return self.encoded_labels

    def normalize_features(self):
        self.normalized_features  = normalize(self.features, axis=0, norm='max')
        self.are_features_normalized = True

    def get_normalized_features(self):
        return self.normalized_features




