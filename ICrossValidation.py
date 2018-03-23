from abc import ABCMeta, abstractmethod

class ICrossvalidation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_data(self, _data): raise NotImplementedError

    @abstractmethod
    def set_model(self, _model): raise NotImplementedError

    @abstractmethod
    def __get_parameters_from_model(self): raise NotImplementedError

    @abstractmethod
    def generate_validation_training_sets(self, **kwargs): raise NotImplementedError

    @abstractmethod
    def set_parameters(self, **kwargs): raise NotImplementedError


from sklearn.model_selection import KFold
import numpy as np


class Crossvalidation(ICrossvalidation):
    """Implements the crossvalidation interface. Serves as the link between neural network and the data it is
    being trained on
    """

    def __init__(self, _data=None, _model=None):
        """

        :param _data:
        :param _model:
        """
        self.data = _data
        self.model = _model
        self.__trained_model_parameters = []

    def set_parameters(self, _folds):
        self.folds = _folds

    def set_model(self, _model):
        self.model = _model

    def set_data(self, _data):
        self.data = _data

    def generate_validation_training_sets(self):
        self.set_parameters(5)
        kf = KFold(n_splits=self.folds)
        self.folds_indexes = kf.split(self.data.features)


    def __get_parameters_from_model(self):
        self.__trained_model_parameters = self.model.get_trained_model_parameters()

    def train_model(self):
        for [train, test], i in zip(self.folds_indexes, range(self.folds)):
            train_features, train_labels = self.__get_features_and_labels(train)
            test_features, test_labels = self.__get_features_and_labels(test)
            self.model.initialize_network(train_features, train_labels)
            #self.model.train_network()
            #self._compare_results_with_training_labels(labels)
            #self.__trained_model_parameters.append(self.__get_parameters_from_model(self.model))
        pass

    def __get_features_and_labels(self, _iteration):
        return [self.data.get_features()[_iteration], self.data.get_labels()[_iteration]]

    def _compare_results_with_training_labels(self, labels):
        normalized_result = np.array(self.model.output)
        comparison_table = []
        for row_train, row_ref in zip(normalized_result, labels):
            if np.array_equal(row_ref, row_train):
                comparison_table.append(1)
            else:
                comparison_table.append(0)
        comparison_table = np.array(comparison_table).reshape((-1, 1))
        self.accuracy = sum(comparison_table) / len(comparison_table)
        print('Accuracy is: '.format(str(self.accuracy)))





