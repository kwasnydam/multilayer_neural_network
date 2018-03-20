import numpy as np
import Synapse
import random

class Neuron:
    """
    Neuron consists of:

    :param input_synapses - list of synapses connected to this neuron (input neurons)
    :param output_synapses - list of synapses(neurons) this neuron is being an input to
    :param sum - sum of input*weigh for every input neuron
    :param output - sigmoidal function of the sum, f(sum)
    """

    BETA = 1
    BIAS = 0

    def __init__(self, sum=0, delta=0, output=0,
                 input_synapses=None, output_synapses=None):
        self.sum = sum
        self.delta = delta
        self.output = output
        self.input_synapses = list()
        self.output_synapses = list()
        #self.bias = random.random()*(-5)#Neuron.BIAS

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

        self.sum = sum #+ self.bias

    def calc_output(self):
        """Output as a sigmoidal function of sum"""
        if self.sum is None:
            return
        #self.output = (2/(1+np.exp((-Neuron.BETA)*self.sum))) - 1
        self.output = (1 / (1 + np.exp((-Neuron.BETA) * self.sum)))
