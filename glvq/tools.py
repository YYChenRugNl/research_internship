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
