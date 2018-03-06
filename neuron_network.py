import neuron_layer

class NeuronNetwork:

    def __init__(self, _neuron_layers_list=None, training_input=None, training_output=None):
        self.neuron_layers_list = _neuron_layers_list
        self.training_input = training_input
        self.training_output = training_output

    @property
    def neuron_layers_list(self):
        if self.neuron_layers_list is not None:
            return self.__neuron_layers_list
        else:
            print('there is no layers in {}',format(self.__name__))

    @neuron_layers_list.setter
    def neuron_layers_list(self, list):
        self.__neuron_layers_list = list

    @property
    def training_input(self):
        if self.training_input is not None:
            return self.__training_input
        else:
            print('there is no training data in {}', format(self.__name__))

    @training_input.setter
    def training_input(self, data):
        self.__training_input = data

    def create_network(self, num_of_layers, neurons_in_each_layer):
        create_neurons(num_of_layers, neurons_in_each_layer)
        create_synapses()
