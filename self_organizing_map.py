from som_neuron import SOMNeuron
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


class SelfOrganizingMap:
    """
     This class represents the Self orgainizng map created in accordance with:
     http://home.agh.edu.pl/~horzyk/lectures/ai/SztucznaInteligencja-SOM.pdf
     It consists of two nested classes, trainer, that is responsible for the training process of SOM
     and Predictor, that produces an output based on the input vector. SOM is an example of unsupervised
     learning method used to organizing a data into a clusters. The goal is that every class from the input
     data set should be represented as a region on the grid or in some other than 2D space of the map.
     The number of nodes the map consist of should be signifocantly lower then the number of inputs so that
     it generalizes well
    """

    def __init__(self, dimensions):
        try:
            if dimensions:
                self.dimensions = dimensions
                self.map = np.array(self._initialize(self.dimensions))
            else:
                raise Exception('Creation error: dimensions badly specified\n')
        except Exception as e:
            print(str(e.args))

    def _initialize(self, dimensions):
        """Creates grid filled with SOMNeurons. Only 2D and 3D map considered"""
        if len(dimensions) == 2:
            map = [[SOMNeuron() for i in range(dimensions[0])] for j in range(dimensions[1])]
        elif len(dimensions) == 3:
            map = [[[SOMNeuron() for i in range(dimensions[0])] for j in range(dimensions[1])] for k in range(dimensions[3])]
        else:
            print('Map initilization error')
            return -1
        return map

    def set_input_len(self, length):
        self.len = length

    def set_input_range(self, range):
        self.input_value_range = range

    def run(self, data, labels):
        self.trainer = SelfOrganizingMap.Trainer(self, data, labels)
        self.trainer.train()

    class Trainer:
        def __init__(self, model, training_data, labels_for_presentation):
            self.model = model
            self.training_data = training_data
            self.pres_labels = labels_for_presentation
            self.winning_neuron = [0 for i in range(len(self.model.dimensions))]
            self.initialize_weights()
            self.__initial_update_radius = max(self.model.dimensions)
            self.__initial_update_rate = 1
            self.__narrowing_constant = 1000

        def initialize_weights(self):
            dim = self.model.dimensions
            map = self.model.map
            initialize_weight = lambda x: x.initialize(self.model.len)
            [initialize_weight(node) for node in self.__get_map_element(map, dim)]
            # if len(dim) > 3:
            #     print('SelfOrganizingMap.Trainer.initialize_weights: ERROR - bad dimensions, sholud never get here')
            # elif len(dim) == 2:
            #     for x, y in itertools.product(range(dim[0]), range(dim[1])):
            #         map[x, y].initialize(self.model.len)
            # else:
            #     for x, y, z in itertools.product(range(dim[0]), range(dim[1]), range(dim[2])):
            #         map[x, y, z].initialize(self.model.len)

        def __get_map_element(self, map, dimension, get_index=False):
            '''
            Generator function that yields map elements
            :param map:         the map which elements should be returned
            :param dimension:   array-like with dimensions of the map, indexed with []
            :param get_index:   if True, the function return a tuple (node, coordinates of the node in the map)
            :return: yields node elements
            '''
            if len(dimension) > 3:
                print('SelfOrganizingMap.Trainer.initialize_weights: ERROR - bad dimensions, sholud never get here')
            elif len(dimension) == 2:
                for x, y in itertools.product(range(dimension[0]), range(dimension[1])):
                    if not get_index:
                        yield map[x, y]
                    else:
                        yield {'node': map[x,y], 'index': tuple([x, y])}
            else:
                for x, y, z in itertools.product(range(dimension[0]), range(dimension[1]), range(dimension[2])):
                    if not get_index:
                        yield map[x, y, z]
                    else:
                        yield {'node': map[x,y, z], 'index': tuple([x, y, z])}

        def train(self):
            iter_count = 0
            update_radius= 1
            update_rate = 1
            #for i in range(200):
            while True:
                iter_count += 1
                update_radius = self.__initial_update_radius*np.exp(-(iter_count/self.__narrowing_constant))
                update_rate = self.__initial_update_rate*np.exp(-(iter_count/self.__narrowing_constant))
                for sample, label in zip(self.training_data, self.pres_labels):
                    self.calc_distances(sample)             # works
                    winner_coordinates = self.find_winner() # works
                    self.model.map[winner_coordinates].winners += label
                    self.adjust_weights(winner_coordinates, update_radius, update_rate)
                self.calc_colors()
                self.show_grid()

        def calc_distances(self, sample):
            """Calculates distance from the input vector for every node"""
            map = self.model.map
            calc_distance = lambda x: x.calculate_distance_from_sample(sample)
            [calc_distance(node) for node in self.__get_map_element(map, self.model.dimensions)]

        def find_winner(self):
            """
            Return a tup-le containing coordinates of the winner(node with the lowest distance from the
            input vector. In case multiple nodes have the exact some distance, only the first found id returned.
            In case a tuple of results cannot be created, it throws an exception that is handled(xD) internally
            """
            distance_matrix = np.array([node.distance for node in self.__get_map_element(
                self.model.map,
                self.model.dimensions
            )])
            distance_matrix = np.reshape(distance_matrix, [dim for dim in self.model.dimensions])
            winner_coords = np.where(distance_matrix == distance_matrix.min())
            winner_coords = [winner_coords[x][0] for x in range(len(self.model.dimensions))]
            try:
                return tuple([i.item() for i in winner_coords])
            except:
                print('cos sie zepsulo, nie powinno tutaj trafic xD')

        def adjust_weights(self, winner_coordinates, update_radius, update_rate):
            map = self.model.map
            [node['node'].adapt_weights(
                SOMNeuron.calculate_distance_coeff(
                    node['index'],
                    winner_coordinates,
                    update_radius
                ), update_rate)
                for node in self.__get_map_element(map, self.model.dimensions, get_index=True)]

        def calc_colors(self):
            map = self.model.map
            [node.calculate_color() for node in self.__get_map_element(map, self.model.dimensions)]

        def show_grid(self):
            """On every loop iteration shows the current state of the grid
            (a heatmap with values depending on how many times a certain node won)"""
            # Set-up the figure to be drawn on
            plt.figure(1)
            plt.ion()
            plt.show()
            colors = [node.color for node in self.__get_map_element(self.model.map,
                                                                    self.model.dimensions)]
            colors = np.reshape(colors, [4, 4])
            sns.heatmap(colors)
            plt.pause(.001)
            plt.clf()       # Clears surface of the whole figure so it can be updated in the next iteration

    class Predictor:
        ''' Contains model reference and method that generates a prediction
        It is assumed that model is trained, i.e. contains a grid made of som_neurons
        that have properly adjusted weights. Input data that the prediction will be made on
        must have the same size as the data the model has been trained on
        '''

        def __init__(self, model):
            self.model = model

        def predict(self, data):















