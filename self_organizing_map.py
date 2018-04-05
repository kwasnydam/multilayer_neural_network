from som_neuron import SOMNeuron

class SelfOrganizingMap:

    def __init__(self, dimensions):
        try:
            if dimensions:
                self.dimensions = dimensions
                self.map = self._initialize(self.dimensions)
            else:
                raise Exception('Creation error: dimensions badly specified\n')
        except Exception as e:
            print(str(e.args))

    def _initialize(self, dimensions):
        """Creates grid filled with SOMNeurons"""
        if len(dimensions) == 2:
            map = [[SOMNeuron() for i in range(dimensions[0])] for j in range(dimensions[1])]
        elif len(dimensions) == 3:
            map = [[[SOMNeuron() for i in range(dimensions[0])] for j in range(dimensions[1])] for k in range(dimensions[3])]
        else:
            print('Map initilization error')
            return -1
        return map

    def set_input_len(self, length):
        self.len = length

    def set_input_range(self, range):
        self.input_value_range = range

    class Trainer:
        def __init__(self, model, training_data):
            self.model = model
            self.training_data = training_data
            self.winning_neuron = [0 for i in range(len(self.model.dimensions))]

        def train(self):
            for sample in self.training_data:
                self.calc_dist









