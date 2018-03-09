import numpy as np


class Neuron:
    """
    Neuron consists of:
    :param
    input_synapses - list of synapses connected to this neuron (input neurons)
    output_synapses - list of synapses(neurons) this neuron is being an input to
    sum - sum of input*weigh for every input neuron
    output - sigmoidal function of the sum, f(sum)
    """

    def __init__(self, sum=0, delta=0, output=0,
                 input_synapses=list(), output_synapses=list()):
        self.sum = sum
        self.delta = delta
        self.output = output
        self.input_synapses = input_synapses
        self.output_synapses = output_synapses

    @property
    def delta(self):
        return self.__delta

    @delta.setter
    def delta(self, _delta):
        self.__delta = _delta

    def calc_sum(self):
        """sum(input*weight) for every input"""
        sum = 0
        for synapse in self.input_synapses:
            sum += synapse.get_value()*synapse.weight
        self.sum = sum

    def calc_output(self):
        """Output as a sigmoidal function of sum"""
        if self.sum is None:
            return
        self.output = 1/(1+np.exp(self.sum))
