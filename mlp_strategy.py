from ml_model_strategy import ModelStrategy
from neuron_network import NeuronNetwork

class MLPStrategy(ModelStrategy):

    def __init__(self):
        self.model = NeuronNetwork()


    def train(self,  training_features, training_labels):
        self.model.train(training_features, training_labels)

    def create(self, **parameters):
        if parameters is not None:
            mlp_no_of_layers = parameters['no_of_layers']
            mlp_size_of_each_layer = parameters['size_of_each_layer']
            self.model.create_network(mlp_no_of_layers, mlp_size_of_each_layer)
        else:
            print('ERROR: Couldnt create a network')

    def predict(self, data):
        return self.model.predict(data)

    def reset(self):
        self.model.reset()

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)


    def get_parameters(self):
        return self.model.get_parameters()