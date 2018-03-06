
class Neuron:
    """
    This is a neuron class
    """

    def __init__(self, sum=0, delta=0, output=0, input_connections=list(),output_connections=list()):
        self.sum = None
        self.delta = None
        self.output = None
        self.input_connections = input_connections
        self.output_connections = output_connections

