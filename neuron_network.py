import neuron_layer as nl
import Synapse

class NeuronNetwork:

    MIU = 1
    THRESHOLD = 1

    def __init__(self, _neuron_layers_list=None, _training_input=None,
                 _training_output=None, _input_layer=None):
        self.neuron_layers_list = _neuron_layers_list
        self.training_input = _training_input
        self.training_output = _training_output
        self.input_layer = _input_layer

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

    @property
    def training_output(self):
        if self.training_output is not None:
            return self.__training_output
        else:
            print('there is no training data in {}', format(self.__name__))

    @training_output.setter
    def training_output(self, data):
        self.__training_output = data

    #def create_input_layer(self, _no_of_neurons):
     #   __input_layer = nl.NeuronLayer(_no_of_neurons)
    @classmethod
    def _create_layers(self, _num_of_layers, _size_of_each_layer):
        __layers = list()
        for layer in range(_num_of_layers):
            #for size_of_layer in _size_of_each_layer:
            __layers.append(nl.NeuronLayer(_size_of_each_layer[layer]))
        return __layers

    def create_network(self, _num_of_layers, _neurons_in_each_layer):
        self.neuron_layers_list = NeuronNetwork._create_layers(_num_of_layers, _neurons_in_each_layer)
        self._wire_layers()

    def initialize_network(self):
        if self.training_input is not None and self.training_output is not None:
            for i_neuron in self.neuron_layers_list[0]:
                for input_data in self.training_input[0]:
                    _i_synapse = Synapse.Synapse(_input=input_data, _out=i_neuron,
                                                 _weight=1, _mode='data')
                    i_neuron.input_synapses.append(_i_synapse)
            self.train()
        else:
            print("Missing Input or Output data")

    def _wire_layers(self):
        for iterator in range(len(self.neuron_layers_list)-1):
            for o_neuron in self.neuron_layers_list[iterator+1]:
                for i_neuron in self.neuron_layers_list[iterator]:
                    _i_synapse = Synapse.Synapse(_input=i_neuron, _out=o_neuron)
                    o_neuron.input_synapses.append(_i_synapse)
                    i_neuron.output_synapses.append(_i_synapse)

    def train(self):

        while True:
            # Input Propagation phase
            for layer in self.neuron_layers_list:
                for neuron in layer:
                    neuron.calc_sum()   # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()

            # Backpropagation phase
            output_layer = self.neuron_layers_list[-1]
            error = 0
            for i in range(len(output_layer)):
                output_layer[i].delta = output_layer[i].output - self.training_output[i]
                error += output_layer[i].delta**2
            if error < NeuronNetwork.THRESHOLD:
                break

            for layer in reversed(self.neuron_layers_list):
                for neuron in layer:
                    for input_synapse in neuron.input_synapses:
                        adjustment = (-1)*NeuronNetwork.MIU*neuron.delta*(1-neuron.output)*neuron.output*input_synapse.get_value()
                        delta =
                        input_synapse.weight_adjustment.append(adjustment)







