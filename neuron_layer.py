import Neroun, Synapse

class NeuronLayer:

    def __init__(self, _size=None):
        self.neuron_list = list()
        if _size is not None:
            self.size = size
            self.create_layer(size)
        self.weight_adjustements

    def create_layer(self, number_of_neurons):
        for i in range(number_of_neurons):
            self.neuron_list.append(Neroun.Neuron())