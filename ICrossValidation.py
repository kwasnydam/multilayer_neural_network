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
        kf = KFold(n_splits=self.folds)
        self.folds_indexes = kf.split(self.data.features)

    def __get_parameters_from_model(self):
        self.__trained_model_parameters = self.model.get_trained_model_parameters()

    def train_model(self):
        for iteration in self.folds:
            features, labels = self.__get_features_and_labels(iteration)
            self.model.initialize_network(features, labels)
            self.model.train_network()
            self.__trained_model_parameters.append(self.__get_parameters_from_model(self.model))
        pass

    def __get_features_and_labels(self, _iteration):
        return [self.data.features[_iteration], self.data.labels[_iteration]]





