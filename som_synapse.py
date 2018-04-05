import numpy as np
from Synapse import Synapse


class SOMSynapse(Synapse):

    def __init__(self, input, output):
        super(Synapse).__init__(input, output)