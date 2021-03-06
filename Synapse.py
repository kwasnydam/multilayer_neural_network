import random
import numpy as np
class Synapse:
    """A synapse class. Synapse consists of input neuron, output_neuron and connection weight"""

    def __init__(self, _input, _out, _weight=0.2, _mode='normal'):
        self.input = _input
        self.out = _out
        #self.weight = _weight
        self.weight = (np.random.rand()) #initailzing weights
        self.weight_adjustment = list()
        self.mode = _mode
        #print('SYNAPSE_INIT: Created synapse with input: {} and output: {}'.format(self.input, self.out))

    def get_value(self):
        if self.mode == 'normal':
            return self.input.output
        else:
            return self.input

    def reset(self):
        self.weight = (np.random.rand())

