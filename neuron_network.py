import neuron_layer as nl
import Synapse
import Neroun
import numpy as np
import os
from io import open
import math


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
        :param _training_input: training feature array
        :param _training_output:training reference labels(classes
        """
        self.training_input = _training_input
        self.training_output = _training_output
        if self.training_input is not None and self.training_output is not None:
            #: Connect the input layer's neurons with the input data
            for i_neuron in self.neuron_layers_list[0]:
                for input_data in self.training_input[0]:
                    _i_synapse = Synapse.Synapse(_input=input_data, _out=i_neuron,
                                                 _weight=1, _mode='data')
                    i_neuron.input_synapses.append(_i_synapse)
        else:
            print("Missing Input or Output data")

    def train(self, _training_features, _training_labels):
        self.initialize_network(_training_features, _training_labels)
        self.train_network()

    def train_network(self):
        self.trainer = NeuronNetwork.Trainer(self, 1024)  #: This object is responsible for training of the network
        self.trainer.train()

    def predict(self, features):
        predictor = NeuronNetwork.Predictor(self, features)
        results = predictor.predict()
        return results

    def get_parameters(self):
        """It is intended to return a list of layers containing list of neurons containg list of synapses weights"""
        output = []
        for layer in self.neuron_layers_list:
            _l = []
            for neuron in layer:
                _n = []
                for synapse in neuron.input_synapses:
                    _n.append(synapse.weight)
                _l.append(_n)
            output.append(_l)
        return output

    def set_parameters(self, parameters):
        for network_layer, parameter_layer in zip(self.neuron_layers_list, parameters):
            for neuron_layer, parameter_neuron in zip(network_layer, parameter_layer):
                for synapse, synapse_weight in zip(neuron_layer.input_synapses, parameter_neuron):
                    synapse.weight = synapse_weight

    def reset(self):
        # Reseting weights
        for layer in self.neuron_layers_list:
            for neuron in layer:
                for synapse in neuron.input_synapses:
                    synapse.reset()
        # breaking the connection to the inputs (it is reasonable because number of inputs may be different)
        for neuron in self.neuron_layers_list[0]:
            neuron.input_synapses = []
        # reseting miu
        self.miu = NeuronNetwork.MIU


    class Trainer:
        """An inner class that is responsible for the training of Neural Network.
        Performs Backpropagation algorithm to adjust the parameters
        """
        THRESHOLD = 0.000001
        def __init__(self, network, max_iter):
            """Creates the Trainer object with reference to the neuron network and a max number of iterations

            Attributes:
            :param network:     :obj:'NeuronNetwork' reference to the NeuronNetwork object
            :param max_iter:    :obj:'int' number of iterations to train the network
            """
            self.network = network      # Reference to the NeuronNetwork object
            self.training_size = len(self.network.training_input[:, 1]) # size of training data (number of samples)
            self.training_output = []   # Predicted labels
            self.max_iter = max_iter    # number of training iterations
            self._set_initial_training_parameters()

            self.is_online = True

        def _set_initial_training_parameters(self):
            """Zero out the error and set the current iteration to be equal 0"""
            self.error = [0]*self.training_size
            self.iteration = 0
            self.previous_error = 9999  # Initial error
            self.network.miu = NeuronNetwork.MIU

        def train(self):
            """Perform the backpropagation algorithm to train the network. Finish criterium is:
            1. Error adjustement in consequetive iterations below given THRESHOLD
            2. Reached maximum number of iterations
            """
            #self.set_initial_training_parameters()
            while self.iteration < self.max_iter:
                # in each iteration
                for sample_in, sample_out in zip(self.network.training_input, self.network.training_output):
                    # Calculate error and weight adjustements for every sample in input data
                    self._connect_inputs(sample_in)
                    self._propagate_forward()
                    self._calculate_error(sample_out)
                    self._propagate_back()
                if self._is_finish_criterium_met():
                    # if the error difference in the current iteration is lower then threshol, break cause minimum reached
                    break
                else:
                    # Else calculate network's parameters' adjustements and restart the state
                    self._caluclate_average_adjustment()
                    self._save_current_error()
                    self._clear_training_process_parameters()
                    #self._adjust_miu()
                    self._increase_iteration()

        def _adjust_miu(self):
            self.network.miu = self.network.miu*math.exp(-10*self.iteration/self.max_iter)
            pass

        def _connect_inputs(self, sample):
            """Feed the current input sample to the netowrk"""
            for i_neuron in self.network.neuron_layers_list[0]:
                for index in range(len(sample)):
                    i_neuron.input_synapses[index].input = sample[index]

        def _propagate_forward(self):
            """Forward propagation phase. Based on current parameters the output is being calculated"""
            for layer in self.network.neuron_layers_list:
                for neuron in layer:
                    neuron.calc_sum()     # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()  # sigmoidal function of the sum
                    neuron.delta = 0

        def _calculate_error(self, output):
            """Calculate the difference between reference, real labels and the calculated outputs of the network"""
            output_layer = self.network.neuron_layers_list[-1]
            error = 0
            for i in range(len(output_layer)):
                output_layer[i].delta = output_layer[i].output - output[i]
                error += output_layer[i].delta ** 2
            self.error.append(error)
            self.training_output.append([output_layer[i].output for i in range(len(output_layer))])

        def _propagate_back(self):
            """Backpropagation of errors using backpropagation algorithm. Adjustements of weigths for the current sample
            are stpred in the synapse and averaged across all samples when running out of data
            """
            for layer in reversed(self.network.neuron_layers_list):
                for neuron in layer:
                    for input_synapse in neuron.input_synapses:
                        adjustment = (-1)*self.network.miu*neuron.delta*(1-neuron.output) *\
                                     neuron.output*input_synapse.get_value()
                        input_synapse.weight_adjustment.append(adjustment)
                        if type(input_synapse.input) is Neroun.Neuron:
                            delta = neuron.delta * input_synapse.weight * (1 - neuron.output) * neuron.output
                            input_synapse.input.delta += delta

                        if self.is_online:
                            input_synapse.weight += adjustment

        def _is_finish_criterium_met(self):
            """Checks whether the error adjustement is lower then the threshold

            :return: True if error adjustement lower then THRESHOLD, false otherwise
            """
            _state = (self.previous_error - (sum(self.error)/len(self.error))) < type(self).THRESHOLD
            self.previous_error = sum(self.error)/len(self.error)
            return _state
            #return False

        def _caluclate_average_adjustment(self):
            """Average obtained weight adjustements and adjust weights accordingly. Clear list of adjustement for next iteration"""
            # print("caluclate_average__adjustment: Hello there")
            for layer in self.network.neuron_layers_list:
                for neuron in layer:
                    for input_synapse in neuron.input_synapses:
                        _adjustement = sum(input_synapse.weight_adjustment) / len(input_synapse.weight_adjustment)
                        input_synapse.weight += _adjustement
                        input_synapse.weight_adjustment = []

        def _clear_training_process_parameters(self):
            """Clears the temporary parameters for next iteration"""
            self.error = [0] * self.training_size
            if self.iteration < self.max_iter-1:
                self.training_output = []

        def _save_current_error(self):
            """Saves current error or comparison in the next itearation"""
            self.previous_error = sum(self.error) / len(self.error)

        def _increase_iteration(self):
            #print(str(self.iteration))
            self.iteration += 1

    class Predictor():

        def __init__(self, model, data):
            self.network = model
            self.data = data
            self.prediction = []

        def predict(self):
            for sample in self.data:
                # Predict output based on the input
                self._connect_inputs(sample)
                self._propagate_forward()
                self._save_results()
            self.output = self._normalize_results()
            return self.output

        def _connect_inputs(self, sample):
            """Feed the current input sample to the network"""
            for i_neuron in self.network.neuron_layers_list[0]:
                for index in range(len(sample)):
                    i_neuron.input_synapses[index].input = sample[index]

        def _propagate_forward(self):
            """Forward propagation phase. Based on current parameters the output is being calculated"""
            for layer in self.network.neuron_layers_list:
                for neuron in layer:
                    neuron.calc_sum()     # calc_sum -> for inp in self.in_syn: self.sum += inp.val*inp.weight
                    neuron.calc_output()  # sigmoidal function of the sum
                    neuron.delta = 0

        def _save_results(self):
            """Append the result of the current predciton to the predicitions list"""
            output_layer = self.network.neuron_layers_list[-1]
            self.prediction.append([output_layer[i].output for i in range(len(output_layer))])

        def _normalize_results(self):
            normalized_results = []
            for result in self.prediction:
                max_val = 0
                result_row = [0 for i in range(len(result))]
                for i in range(len(result)):
                    if result[i] > max_val:
                        max_val = result[i]
                        max_val_index = i
                result_row[max_val_index] = 1
                normalized_results.append(result_row)
            return normalized_results




        '''def _compare_results_with_training_labels(self):
            # IT SHOULD GO TO THE CROSSVALIDATION CLASS
            results = self.training_output
            reference = self.network.training_output
            normalized_result = []
            for result in results:
                if result[0]==max(result):
                    normalized_result.append([1, 0, 0])

                elif result[1] == max(result):
                    normalized_result.append([0, 1, 0])
                else:
                    normalized_result.append([0, 0, 1])

            normalized_result = np.array(normalized_result)
            wynik = []
            for row_train, row_ref in zip(normalized_result, reference):
                if np.array_equal(row_ref, row_train):
                    wynik.append(1)
                else:
                    wynik.append(0)
            wynik = np.array(wynik).reshape((-1, 1))
            self.accuracy = sum(wynik)/len(wynik)
            print('Accuracy is: '.format(str()self.accuracy))'''


'''
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
                    input_synapse.weight_adjustment = []'''









