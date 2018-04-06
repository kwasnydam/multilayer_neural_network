from som_neuron import SOMNeuron
import numpy as np
import itertools

class SelfOrganizingMap:

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
        """Creates grid filled with SOMNeurons"""
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
            map = self.model.map
            calc_distance = lambda x: x.calculate_distance_from_sample(sample)
            [calc_distance(node) for node in self.__get_map_element(map, self.model.dimensions)]

        def find_winner(self):
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
                print('cos sie zjebalo')

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
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import time
            color_map = [[[] for x in range(self.model.dimensions[0])] for y in range(self.model.dimensions[1])]
            [color_map[node['index'][0]][node['index'][1]].append(colors.to_rgb(node['node'].color)) for node in self.__get_map_element(
                self.model.map,
                self.model.dimensions,
                get_index=True)]
            color_map = np.array(color_map)
            my_cmap = colors.LinearSegmentedColormap('my_colormap', color_map, 256)
            # color_map = np.reshape(color_map,
            #                        [dim for dim in self.model.dimensions])
            plt.imshow(color_map, cmap=my_cmap)
            time.sleep(0.001)

















