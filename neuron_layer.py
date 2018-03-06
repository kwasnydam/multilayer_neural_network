import Neroun, Synapse

class NeuronLayer:

    def __init__(self, _neuron_list=list()):
        self.neuron_list = _neuron_list

    def create_layer(self, number_of_neurons):
        for i in range(number_of_neurons):
            self.neuron_list.append(Neroun.Neuron())