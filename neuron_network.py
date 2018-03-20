import neuron_layer as nl
import Synapse
import Neroun
import numpy as np
import os
from io import open


class NeuronNetwork:
    """The NeuronNetwork class is responsible for creating, training and running our neural network

    Attributes:
        neuron_layers_list (list of neuron_layers): A list of layers our net is built of
        training_input:     Matrix of features
        training_output:    Matrix of class labels
        network_output:     Results of running a neural network
    """
    MIU = 1
    THRESHOLD = 0.2

    def __init__(self):
        """
        creates a NeuronNetwork instance. In order to initailze the network and start the training
        one must call the 'initialize_network' method
        """
        self.neuron_layers_list = None  #: A list of layers our net is built of
        self.training_input = None      #: Training input data (matrix of features)
        self.training_output = None     #: Training output data (labels)
        self.input_layer = None         #: I think it is generally unused
        self.miu = NeuronNetwork.MIU    # The rate of learning parameter
        self.network_output = []        #: Classification results of our neuron network
        self.trainer = NeuronNetwork.Trainer(self)  #: This object is responsible for training of the network

    @property
    def neuron_layers_list(self):
        """
        A list containing NeuronLayer objects
        :return: A list containing NeuronLayer objects
        """
        if self.__neuron_layers_list is not None:
            return self.__neuron_layers_list
        else:
            print('there is no layers in {}'.format(self.__name__))

    @neuron_layers_list.setter
    def neuron_layers_list(self, _list):
        self.__neuron_layers_list = _list

    @property
    def training_input(self):
        """Training input data"""
        if self.__training_input is not None:
            return self.__training_input
        else:
            print('there is no training data in {}', format(self.__name__))

    @training_input.setter
    def training_input(self, data):
        self.__training_input = data

    @property
    def training_output(self):
        """Training output data (labels/classes)"""
        if self.__training_output is not None:
            return self.__training_output
        else:
            print('there is no training data in {}', format(self.__name__))

    @training_output.setter
    def training_output(self, data):
        self.__training_output = data

    def create_network(self, _num_of_layers, _neurons_in_each_layer):
        """Creates a network with desired topology. First creates a layers with neurons and then wires them wi

        :param _num_of_layers:  :obj:'int' number of NeuronLayer objects to be created
        :param _neurons_in_each_layer:  :obj:'list' of :obj:'int' number of Neuron objects in each layer
        """
        self.neuron_layers_list = NeuronNetwork._create_layers(_num_of_layers, _neurons_in_each_layer)
        self._wire_layers()

    @classmethod
    def _create_layers(self, _num_of_layers, _size_of_each_layer):
        """Takes number of layers and size of each layer and returns a list of NeuronLayer objects

        Args:
            :param _num_of_layers:      :obj:'int' number of desired layers
            :param _size_of_each_layer: :obj:'list' of :obj:'int' number of neurons in each layer
            :return:                    :obj:'list' of :obj: 'NeuronLayer'
        """
        __layers = list()
        for layer in range(_num_of_layers):
            __layers.append(nl.NeuronLayer(_size_of_each_layer[layer]))
        return __layers

    def _wire_layers(self):
        """
        Being called when a network is already created. Wires Neuron objects in each NeuronLayer object
        with Neuron objects in the next layer with synapse objects.
        :return:
        """
        for iterator in range(len(self.neuron_layers_list)-1):
            for o_neuron in self.neuron_layers_list[iterator+1]:
                for i_neuron in self.neuron_layers_list[iterator]:
                    _i_synapse = Synapse.Synapse(_input=i_neuron, _out=o_neuron)    # Create synapse with input and output
                    o_neuron.input_synapses.append(_i_synapse)      # connect synapse to output neuron;s input
                    i_neuron.output_synapses.append(_i_synapse)     # connect synapse to input neuron;s output

    def initialize_network(self, _training_input, _training_output):
        """Initialze the network with an input and output data to be trained on
        :param _training_input:
        :param _training_output:
        """
        self.training_input = _training_input
        self_training_output = _training_output
        if self.training_input is not None and self.training_output is not None:
            #: Connect the input layer's neurons with the input data
            for i_neuron in self.neuron_layers_list[0]:
                for input_data in self.training_input[0]:
                    _i_synapse = Synapse.Synapse(_input=input_data, _out=i_neuron,
                                                 _weight=1, _mode='data')
                    i_neuron.input_synapses.append(_i_synapse)
        else:
            print("Missing Input or Output data")

    class Trainer:
        def __init__(self, network):
            self.network = network
            self.training_size = len(self.network.training_input[:, 1])
            self.training_output = []
            self._set_initial_training_parameters()

        def train(self):
            #self.set_initial_training_parameters()
            for sample_in, sample_out in self.network.training_input, self.network.training_output:
                self._connect_inputs(sample_in)
                self._propagate_forward()
                self._calculate_error(sample_out)
                self._propagate_back()


        def _connect_inputs(self, sample):
            for i_neuron in self.network.neuron_layers_list[0]:
                for index in range(len(sample)):
                    i_neuron.input_synapses[index].input = sample[index]

        def _propagate_forward(self):
            for layer in self.network.neuron_layers_list:
                for neuron in layer:
                    neuron.calc_sum()     # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()  # sigmoidal function of the sum
                    neuron.delta = 0

        def _calculate_error(self, output):
            output_layer = self.network.neuron_layers_list[-1]
            error = 0
            for i in range(len(output_layer)):
                output_layer[i].delta = output_layer[i].output - output[i]
                error += output_layer[i].delta ** 2
            self.error.append(error)
            self.training_output.append([output_layer[i].output for i in range(len(output_layer))])

        def _set_initial_training_parameters(self):
            self.error = [0]*self.training_size
            self.iteration = 0

        def _propagate_back(self):
            for layer in reversed(self.network.neuron_layers_list):
                for neuron in layer:
                    for input_synapse in neuron.input_synapses:
                        adjustment = (-1)*self.network.miu*neuron.delta*(1-neuron.output) *\
                                     neuron.output*input_synapse.get_value()
                        input_synapse.weight_adjustment.append(adjustment)
                        if type(input_synapse.input) is Neroun.Neuron:
                            delta = neuron.delta * input_synapse.weight * (1 - neuron.output) * neuron.output
                            input_synapse.input.delta += delta

    def train(self):
        k = 0
        error = [0]*(len(self.training_input[:, 1]))
        iteration = 0
        while iteration<1000:
            for i_neuron in self.neuron_layers_list[0]:
                for index in range(len(self.training_input[k])):
                    i_neuron.input_synapses[index].input = self.training_input[k][index]
            # Input Propagation phase

            for layer in self.neuron_layers_list:
                for neuron in layer:
                    neuron.calc_sum()   # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()
                    neuron.delta = 0

            # Backpropagation phase
            output_layer = self.neuron_layers_list[-1]


            for i in range(len(output_layer)):
                output_layer[i].delta = output_layer[i].output - self.training_output[k][i]
                error[k] += output_layer[i].delta**2
            self.network_output.append([output_layer[i].output for i in range(len(output_layer))])

            for layer in reversed(self.neuron_layers_list):
                for neuron in layer:
                    for input_synapse in neuron.input_synapses:
                        adjustment = (-1)*self.miu*neuron.delta*(1-neuron.output) *\
                                     neuron.output*input_synapse.get_value()
                        input_synapse.weight_adjustment.append(adjustment)
                        if type(input_synapse.input) is Neroun.Neuron:
                            delta = neuron.delta * input_synapse.weight * (1 - neuron.output) * neuron.output
                            input_synapse.input.delta += delta

            if k < len(self.training_input[:, 1])-1:
                k += 1
            else:
                if sum(error)/len(error) < NeuronNetwork.THRESHOLD:
                    break
                ouput = np.array(self.network_output)
                self.caluclate_average__adjustment()
                k = 0
                error = [0] * (len(self.training_input[:, 1]))
                self.network_output = []
                iteration += 1

    def caluclate_average__adjustment(self):
        #print("caluclate_average__adjustment: Hello there")
        for layer in self.neuron_layers_list:
            for neuron in layer:
                for input_synapse in neuron.input_synapses:
                    _adjustement = sum(input_synapse.weight_adjustment)/len(input_synapse.weight_adjustment)
                    input_synapse.weight += _adjustement
                    input_synapse.weight_adjustment = []









