import Neroun, Synapse


class NeuronLayer:

    def __init__(self, _size=None):
        self.neuron_list = list()
        if _size is not None:
            self.size = _size
            self.create_layer(_size)
            self.__counter = -1

    def create_layer(self, number_of_neurons):
        for i in range(number_of_neurons):
            self.neuron_list.append(Neroun.Neuron())

    def __getitem__(self, index):
        return self.neuron_list[index]

    def __len__(self):
        return len(self.neuron_list)
