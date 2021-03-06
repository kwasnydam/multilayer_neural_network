import neuron_layer as nl
import Synapse
import Neroun
import os


class NeuronNetwork:

    MIU = 0.99
    THRESHOLD = 0.5

    def __init__(self, _neuron_layers_list=None, _training_input=None,
                 _training_output=None, _input_layer=None):
        self.neuron_layers_list = _neuron_layers_list
        self.training_input = _training_input
        self.training_output = _training_output
        self.input_layer = _input_layer
        self.miu = NeuronNetwork.MIU

    @property
    def neuron_layers_list(self):
        if self.__neuron_layers_list is not None:
            #print('Get_neuron_layers_list')
            return self.__neuron_layers_list
        else:
            print('there is no layers in {}',format(self.__name__))

    @neuron_layers_list.setter
    def neuron_layers_list(self, _list):
        self.__neuron_layers_list = _list

    @property
    def training_input(self):
        if self.__training_input is not None:
            return self.__training_input
        else:
            print('there is no training data in {}', format(self.__name__))

    @training_input.setter
    def training_input(self, data):
        self.__training_input = data

    @property
    def training_output(self):
        if self.__training_output is not None:
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
            #self.train()
        else:
            print("Missing Input or Output data")

    def _wire_layers(self):
        for iterator in range(len(self.neuron_layers_list)-1):
            for o_neuron in self.neuron_layers_list[iterator+1]:
                for i_neuron in self.neuron_layers_list[iterator]:

                    _i_synapse = Synapse.Synapse(_input=i_neuron, _out=o_neuron)
                    #print("WIRE LAYERS: Connecting neuron {} from layer {} with neuron {} from layer {} "
                    #      "using synapse {} with input {} and output {}".format(
                    #    id(i_neuron),
                    #    id(self.neuron_layers_list[iterator]),
                    #    id(o_neuron),
                    #    id(self.neuron_layers_list[iterator + 1]),
                    ##    id(_i_synapse),
                     #   id(_i_synapse.input),
                     #   id(_i_synapse.out)
                     #   ))
                    o_neuron.input_synapses.append(_i_synapse)
                    i_neuron.output_synapses.append(_i_synapse)

    def train(self):
        k = 0
        while True:
            for i_neuron in self.neuron_layers_list[0]:
                #i_neuron.input_synapses = []
                #for input_data in self.training_input[k]:
                for index in range(len(self.training_input[k])):
                    #_i_synapse = Synapse.Synapse(_input=input_data, _out=i_neuron,
                    #                            _weight=1, _mode='data')
                    #print(str(self.training_input[k][index]))
                    i_neuron.input_synapses[index].input = self.training_input[k][index]
            # Input Propagation phase

            for layer in self.neuron_layers_list:
                _log_entry =""
                for neuron in layer:
                    neuron.calc_sum()   # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()
                    neuron.delta = 0
                    _log_entry += "n_o: {}\t".format(neuron.output)
                with open('log_neurons_outputs.txt', mode='a', encoding='utf-8') as myFile:
                      _log_entry += "\n"
                      myFile.write(_log_entry)
                       # print(_log_entry)
            with open('log_neurons_outputs.txt', mode='a', encoding='utf-8') as myFile:
                myFile.write("\n")

            # Backpropagation phase
            output_layer = self.neuron_layers_list[-1]
            error = 0
            _log_entry = ""
            for i in range(len(output_layer)):
                output_layer[i].delta = output_layer[i].output - self.training_output[k][i]
                error += output_layer[i].delta**2
                _log_entry += "neuron {}: {}\t".format(i, output_layer[i].output)

            with open('log_network_outputs.txt', mode='a', encoding='utf-8') as myFile:
                _log_entry += "\n"
                myFile.write(_log_entry)
                print(_log_entry)
                myFile.write("\n"+"{}\t{}\t{}\n".format(self.training_output[k][0], self.training_output[k][1],
                                                 self.training_output[k][2]))
                #_log_entry = ""

            if error < NeuronNetwork.THRESHOLD:
                break

            for layer in reversed(self.neuron_layers_list):
                for neuron in layer:

                    for input_synapse in neuron.input_synapses:
                        adjustment = (-1)*self.miu*neuron.delta*(1-neuron.output) *\
                                     neuron.output*input_synapse.get_value()
                        input_synapse.weight_adjustment.append(adjustment)
                        if type(input_synapse.input) is Neroun.Neuron:
                            delta = neuron.delta * input_synapse.weight * (1 - neuron.output) * neuron.output
                            input_synapse.input.delta += delta
                        #else:


            if k < len(self.training_input[:, 1])-1:
                k += 1
                #print(str(k))
            else:
                self.caluclate_average__adjustment()
                k = 0
                if self.miu > 0.01:
                    self.miu -= 0.001
                else:
                    self.miu = 0.01

    def caluclate_average__adjustment(self):
        #print("caluclate_average__adjustment: Hello there")
        for layer in self.neuron_layers_list:
            for neuron in layer:
                _log_entry = ""
                for input_synapse in neuron.input_synapses:
                    _adjustement = sum(input_synapse.weight_adjustment)/len(input_synapse.weight_adjustment)
                    input_synapse.weight += _adjustement
                    input_synapse.weight_adjustment = []
                    _log_entry += "weight: {}\t".format(input_synapse.weight)
                with open('log_synapses_adjusted_weights.txt', mode='a', encoding='utf-8') as myFile:
                    _log_entry += "\n"
                    myFile.write(_log_entry)
                #print(_log_entry)







