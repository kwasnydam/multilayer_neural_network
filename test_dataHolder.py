from unittest import TestCase
from DataHolder import DataHolder
import unittest
import pandas as pd

class TestDataHolder(TestCase):

    def setUp(self):
        self.filename = './IrisDataTrain.xls'
        self.data_holder = DataHolder(_filename=self.filename, _number_of_fetures=4)

    def tearDown(self):
        pass

    def test_load_data(self):
        self.data_holder.load_data('lalala.xls', 4)
        self.assertRaises(FileNotFoundError)
        self.data_holder.load_data('lalala.xxx', 4)
        self.assertRaises(Exception)
        self.assertIsNotNone(self.data_holder.load_data(self.filename, 4))

    def test_set_features(self):
        self.data_holder.set_features()
        self.assertEqual(len(self.data_holder.features[1,:]), self.data_holder.number_of_features)

    #def test_get_features(self):
    #    self.fail()

    def test_set_labels(self):
        self.data_holder.set_labels()
        self.assertEqual(len(self.data_holder.labels[1, ]), 1, msg='Labels length should be equal 1')

    #def test_get_labels(self):
    #    self.fail()

    def test_encode_labels(self):
        self.data_holder.encode_labels()
        self.assertEqual(len(self.data_holder.encoded_labels[1, :]), 3, msg='Should equal 3 for Iris Data')

'''   def test_get_encoded_labels(self):
        self.fail()

    def test_normalize_features(self):
        self.fail()

    def test_get_normalized_features(self):
        self.fail()
'''

if __name__ == '__main__':
    unittest.main()

###### setUpClass and tearDownClass ######

@classmethod
def setUpClass(cls):
    print('setupClass')

@classmethod
def tearDownClass(cls):
    print('teardownClass')