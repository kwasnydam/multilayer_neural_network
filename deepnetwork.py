from mlp_strategy import MLPStrategy
from som_mlp_strategy import SOM_MLP_Strategy

class DeepNetwork(object):
    '''

    '''

    def __init__(self, strategy):
        self._model_strategy = strategy # implementing strategy pattern so I can use different models with a common interface
        self.results = None
        self.is_trained = False

    def create(self, **parameters):
        self._model_strategy.create(no_of_layers=parameters['no_of_layers'],
                                    size_of_each_layer=parameters['size_of_each_layer'],
                                    som_size=parameters.get('som_size'))

    def train(self, training_features, training_labels):
        self._model_strategy.train(training_features, training_labels)
        self.is_trained = True

    def predict(self, data):
        return self._model_strategy.predict(data)

    def reset(self):
        self._model_strategy.reset()

    def get_parameters(self):
        return self._model_strategy.get_parameters()

    def set_parameters(self, parameters):
        self._model_strategy.set_parameters(parameters)
     
        
class MLPNetwork(DeepNetwork):
    
    def __init__(self):
        super(MLPNetwork, self).__init__(MLPStrategy())

class SomMlpNetwork(DeepNetwork):

    def __init__(self):
        super(SomMlpNetwork, self).__init__(SOM_MLP_Strategy())
