import numpy as np


class Neuron:
    """
    This is a neuron class
    """

    def __init__(self, sum=0, delta=0, output=0, input_synapses=list(), output_synapses=list()):
        self.sum = sum
        self.delta = delta
        self.output = output
        self.input_synapses = input_synapses
        self.output_synapses = output_synapses

    def calc_sum(self):
        sum = 0
        for synapse in self.input_synapses:
            sum += synapse.input*synapse.weight
        self.sum = sum

    def calc_output(self):
        if self.sum is None:
            return
        self.output = 1/(1+np.exp(self.sum))
