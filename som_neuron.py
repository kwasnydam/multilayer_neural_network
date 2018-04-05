import numpy as np


class SOMNeuron:
    """

    """
    def __init__(self, length=0):
        self.weights = []
        self.distance = 0
        self.winners = [0, 0, 0]
        self.colour = [0, 0, 0]
        if length != 0:
            self.initialize()
        else:
            pass

    def initialize(self, length):
        weights = [self._generate_weight() for i in range(length)]
        self.weights = np.array(weights)

    @staticmethod
    def _generate_weight():
        return abs(np.random.randn()/20)

    def calculate_distance_from_sample(self, input_vector):
        inp = np.array(input_vector)
        calc_dist = lambda x, y: np.sqrt(np.sum((x-y)**2))  # Calculate distance
        self.distance = calc_dist(self.weights, inp)
        print('Distance between weights and input is: ' + str(self.distance))

    def calculate_color(self):
        if self.winners:
            winners = self.winners
            return [winners[0], winners[1], winners[2]]*(255/sum(winners))
        else:
            print('Couldnt calculate color: no data in self.winners')

    def adapt_weights(self, distance_to_winner_coeff, adapt_coeff):
        #distance_coeff = self._calculate_distance_coeff()
        new_weight = self.weights + distance_to_winner_coeff*adapt_coeff*self.distance
        self.weights = new_weight

    @staticmethod
    def _calculate_distance_coeff(this_coord, winner_coord, sigma_t):
        return np.exp(-(np.sum((np.array(this_coord) - np.array(winner_coord)) ** 2))/(2*sigma_t**2))






