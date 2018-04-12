from unittest import TestCase
from self_organizing_map import SelfOrganizingMap
from DataHolder import DataHolder


class TestSelfOrganizingMap(TestCase):

    def setUp(self):
        self.tested_object = SelfOrganizingMap([4, 4])
        self.data = DataHolder(_filename='./IrisDataTrain.xls', _number_of_fetures=4)
        self.data.normalize_features()
        self.data.encode_labels()
        self.tested_object.set_input_len(4)

    def tearDown(self):
        pass

    def test__initialize(self):
        map = self.tested_object._initialize([5,5])
        print('test__initialize zakonczony')

    def test_run(self):
        self.tested_object.run(self.data.get_features(), self.data.get_labels())
        print('sialalalala')