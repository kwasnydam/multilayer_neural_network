import abc


class ModelStrategy(object):
    """You do not need to know about metaclasses.
    Just know that this is how you define abstract
    classes in Python."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def train(self, training_features, training_labels):
        """Required Method"""

    @abc.abstractmethod
    def create(self, **parameters):
        """Required Method"""

    @abc.abstractmethod
    def predict(self, data):
        """Required Method"""

    @abc.abstractmethod
    def reset(self):
        """Required Method"""

    @abc.abstractmethod
    def set_parameters(self, parameters):
        """Required Method"""

    @abc.abstractmethod
    def get_parameters(self):
        """Required Method"""
