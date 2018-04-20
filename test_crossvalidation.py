from unittest import TestCase
import unittest
from DataHolder import DataHolder
from ICrossValidation import Crossvalidation
from neuron_network import NeuronNetwork
from deepnetwork import MLPNetwork, SomMlpNetwork

class TestCrossvalidation(TestCase):

    def setUp(self):
        self.filename = './IrisDataTrain.xls'
        self.data = DataHolder(self.filename, 4)
        # self.model = NeuronNetwork()
        # self.model.create_network(4, [10, 7, 4, 3])

        # self.model = MLPNetwork()
        # self.model.create(no_of_layers=4, size_of_each_layer=[4, 10, 7, 3])

        self.model = SomMlpNetwork()
        self.model.create(no_of_layers=4,
                          size_of_each_layer=[4,10,7,3],
                          som_size=[3,3])

        self.crossvali = Crossvalidation()
        self.crossvali.set_data(self.data)
        self.crossvali.set_parameters(5)
        self.crossvali.generate_validation_training_sets()
        self.crossvali.model = self.model
        self.data.encode_labels()
        self.data.normalize_features()




    def tearDown(self):
        pass

    def test_set_parameters(self):
        self.crossvali.set_parameters(5)
        self.assertEqual(self.crossvali.folds, 5)

    def test_set_model(self):
        self.crossvali.set_model(self.model)
        #self.assertIsInstance(self.crossvali.model, NeuronNetwork)

    def test_set_data(self):
        self.crossvali.set_data(self.data)

    def test_generate_validation_training_sets(self):
        self.crossvali.set_data(self.data)
        self.crossvali.generate_validation_training_sets()


    def test_train_model(self):
        self.crossvali.train_model()
'''
    def test__compare_results_with_training_labels(self):
        self.fail()'''

if __name__ == '__main__':
    unittest.main()

###### setUpClass and tearDownClass ######

@classmethod
def setUpClass(cls):
    print('setupClass')

@classmethod
def tearDownClass(cls):
    print('teardownClass')