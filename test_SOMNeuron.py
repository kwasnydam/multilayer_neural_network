from unittest import TestCase
from som_neuron import SOMNeuron
import numpy as np

class TestSOMNeuron(TestCase):
    def setUp(self):
        self.tested_obj = SOMNeuron()

    def tearDown(self):
        pass

    def test_initialize(self):
        self.tested_obj.initialize(6)
        self.assertEqual(
            len(self.tested_obj.weights), 6, "Length of weigths vector is {} should be 6".format(
            len(self.tested_obj.weights))
        )
        print([self.tested_obj.weights])

    def test__generate_weight(self):
        for i in range(6):
            print(self.tested_obj._generate_weight())
        pass

    def test_calculate_distance_from_sample(self):
        self.tested_obj.initialize(3)
        self.tested_obj.calculate_distance_from_sample([0.8, 0.6, 0.5])

    def test_calculate_color(self):
        pass

    def test_adapt_weights(self):
        self.tested_obj.initialize(3)
        self.tested_obj.calculate_distance_from_sample([0.8, 0.6, 0.5])
        self.tested_obj.adapt_weights(
            distance_to_winner_coeff=SOMNeuron._calculate_distance_coeff([0, 0], [1, 1], 0.9),
            adapt_coeff=0.9)

    def test__calculate_distance_coeff(self):
        self.assertEqual(
            np.round(SOMNeuron._calculate_distance_coeff([0, 0], [1, 1], 0.90)),
            np.round(3.41), 'Distance coeff should be equal {a} is {b}'.format(
                a=3.41,
                b=SOMNeuron._calculate_distance_coeff([0, 0], [1, 1], 0.99)
                )
            )
