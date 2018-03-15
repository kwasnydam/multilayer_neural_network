import Neroun, Synapse

class NeuronLayer:

    def __init__(self, _size=None):
        self.neuron_list = list()
        if _size is not None:
            self.size = _size
            self.create_layer(_size)
            self.__counter = -1

        #self.weight_adjustements

    def create_layer(self, number_of_neurons):
        for i in range(number_of_neurons):
            self.neuron_list.append(Neroun.Neuron())

    '''def __iter__(self):
        return self#.neuron_list[self.__counter]

    def __next__(self):

        if self.__counter < len(self.neuron_list)-1:
            self.__counter += 1
            print("ITER_NEXT: Neuron number {} called".format(self.__counter))
            return self.neuron_list[self.__counter]

        else:
            self.__counter = -1
            raise StopIteration'''
    def __getitem__(self, index):
        return self.neuron_list[index]

    def __len__(self):
        return len(self.neuron_list)
