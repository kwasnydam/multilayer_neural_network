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
        self.accuracy_list=[]

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
            self.model.train(train_features, train_labels)
            self.test_trained_model(test_features, test_labels)
            self.__trained_model_parameters.append(self.model.get_parameters())
            self.model.reset()
            #self.model.train_network()
            #self._compare_results_with_training_labels(labels)
            #self.__trained_model_parameters.append(self.__get_parameters_from_model(self.model))
        self.avg_parameters()
        # self.get_trained_model()

    def test_trained_model(self, test_features, test_labels):
        prediction = self.model.predict(test_features)
        accuracy = self.compare(prediction, test_labels)
        self.accuracy_list.append(accuracy)
        print(str(accuracy))

    def __get_features_and_labels(self, _iteration):
        return [self.data.get_features()[_iteration], self.data.get_labels()[_iteration]]

    def compare(self, prediction, test_labels):
        normalized_result = np.array(prediction)
        comparison_table = []
        for row_train, row_ref in zip(normalized_result, test_labels):
            if np.array_equal(row_ref, row_train):
                comparison_table.append(1)
            else:
                comparison_table.append(0)
        comparison_table = np.array(comparison_table).reshape((-1, 1))
        self.accuracy = sum(comparison_table) / len(comparison_table)
        print('Accuracy is: '.format(str(self.accuracy)))
        return self.accuracy

    def avg_parameters(self):
        averaged_parameters = []
        #for parameters_collection in self.__trained_model_parameters:
        for i in range(len(self.__trained_model_parameters[0])):
            _l = []
            for j in range (len(self.__trained_model_parameters[0][0])):
                #for k in range(self.__trained_model_parameters[0][0][0]):
                average_weight = [[]]
                for k in range(len(self.__trained_model_parameters)):
                    for synapse in self.__trained_model_parameters[k][i][j]:
                        average_weight[k].append(self.__trained_model_parameters[k][i][j])
                average_weight_values = [sum(average_weight[k])/len(average_weight[k]) for k in range(len(self.__trained_model_parameters))]
                _l.append(average_weight_values)





       print('bk')





