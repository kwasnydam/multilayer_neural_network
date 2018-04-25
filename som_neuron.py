import numpy as np


class SOMNeuron:
    """

    """
    def __init__(self, length=0):
        self.weights = []
        self.distance = 0
        self.output = self.distance
        self.winners = [0, 0, 0]
        self.color = [0, 0, 0]
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
        self.distance_for_weights = input_vector - self.weights
        self.output = self.distance
        #print('Distance between weights and input is: ' + str(self.distance))

    def calculate_color(self):
        # if sum(self.winners)>0:
        #     winners = self.winners
        #     self.color = [winners[0]*(1/sum(winners)),
        #                   winners[1]*(1/sum(winners)),
        #                   winners[2]*(1/sum(winners))]
        # else:
        #     self.color=[0,0,0]
        #     #print('Couldnt calculate color: no data in self.winners')
        winners = self.winners
        if max(winners) == 0:
            self.color = -2
            return
        if winners[0] == max(winners):
            self.color = -1
        elif winners[1] == max(winners):
            self.color = 0
        else:
            self.color = 1

    def adapt_weights(self, distance_to_winner_coeff, adapt_coeff):
        #distance_coeff = self._calculate_distance_coeff()
        new_weight = self.weights + distance_to_winner_coeff*adapt_coeff*self.distance_for_weights
        self.weights = new_weight

    @staticmethod
    def calculate_distance_coeff(this_coord, winner_coord, sigma_t):
        return np.exp(-(np.sum((np.array(this_coord) - np.array(winner_coord)) ** 2))/(2*sigma_t**2))






