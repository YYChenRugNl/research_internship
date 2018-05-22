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
        return attr, self.relabel_data(labels[:, 0], 10)

    def read_from_file2(self, datapath):
        my_data = genfromtxt(datapath, delimiter=' ')
        arr_length = len(my_data[0, :])
        # print(arr_length)
        attr = my_data[:, 0:arr_length-1]
        labels = my_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return attr, self.relabel_data(labels[:, 0], 10)

    def relabel_data(self, labels, n_classes):
        process_data = labels.copy()
        unit = 100 / n_classes
        current_rank = -1
        rank_begin_label = np.inf
        rank_end_label = np.inf
        for i in range(n_classes):
            # percentile of boundary in the beginning and in the end
            bound_begin = i * unit
            bound_end = (i+1) * unit
            if bound_end > 100:
                bound_end = 100
            label_begin = np.percentile(labels, bound_begin)
            label_end = np.percentile(labels, bound_end)


            if rank_begin_label != label_begin and rank_end_label != label_end:
                current_rank += 1
                rank_begin_label = label_begin
                rank_end_label = label_end

            if label_begin == label_end:
                process_data[labels == label_end] = current_rank
            elif i == 0:
                process_data[(labels <= label_end) & (labels >= label_begin)] = current_rank
            else:
                process_data[(labels <= label_end) & (labels > label_begin)] = current_rank

        return process_data.astype(int)

    def artificial_data(self, sample_size, list_center, list_label, list_matrix, if_normalize=False):
        nb_ppc = sample_size
        k = len(list_center)
        for i in range(k):
            if i == 0:
                toy_label = np.ones(nb_ppc) * list_label[i]
                toy_data = np.random.multivariate_normal(list_center[i], np.array(list_matrix[i]), size=nb_ppc)

            else:
                index = i
                toy_label = np.append(toy_label, np.ones(nb_ppc) * list_label[i], axis=0)
                toy_data = np.append(toy_data,
                                     np.random.multivariate_normal(list_center[index], np.array(list_matrix[index]),
                                                                       size=nb_ppc), axis=0)

        if if_normalize:
            mean0 = toy_data[:, 0].mean()
            std0 = toy_data[:, 0].std()
            toy_data[:, 0] = (toy_data[:, 0] - mean0) / std0

            mean1 = toy_data[:, 1].mean()
            std1 = toy_data[:, 1].std()
            toy_data[:, 1] = (toy_data[:, 1] - mean1) / std1

        return toy_data, toy_label
