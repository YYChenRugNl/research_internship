from numpy import genfromtxt
import numpy as np


class CustomTool():
    def read_from_file(self, datapath, toDelete):
        my_data = genfromtxt(datapath, delimiter=',')
        new_data = np.delete(my_data, toDelete, 1)
        arr_length = len(new_data[0, :])
        # print(arr_length)
        attr = new_data[:, 0:arr_length-1]
        labels = new_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return attr, labels[:, 0]

    def read_from_file2(self, datapath):
        my_data = genfromtxt(datapath, delimiter=' ')
        arr_length = len(my_data[0, :])
        # print(arr_length)
        attr = my_data[:, 0:arr_length-1]
        labels = my_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return attr, labels[:, 0]

    def artificial_data(self, sample_size, list_center, list_matrix):
        nb_ppc = sample_size
        toy_label = np.zeros(nb_ppc)
        toy_data = np.random.multivariate_normal(list_center[0], np.array(list_matrix[0]), size=nb_ppc)
        k = len(list_center)
        if k > 0:
            for i in range(k - 1):
                index = i + 1
                toy_label = np.append(toy_label, np.ones(nb_ppc) * index, axis=0)
                toy_data = np.append(toy_data,
                                     np.random.multivariate_normal(list_center[index], np.array(list_matrix[index]),
                                                                   size=nb_ppc), axis=0)

        return toy_data, toy_label