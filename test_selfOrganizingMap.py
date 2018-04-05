from unittest import TestCase
from self_organizing_map import SelfOrganizingMap


class TestSelfOrganizingMap(TestCase):

    def setUp(self):
        self.tested_object = SelfOrganizingMap([5, 5])

    def tearDown(self):
        pass

    def test__initialize(self):
        map = self.tested_object._initialize([2, 2])
