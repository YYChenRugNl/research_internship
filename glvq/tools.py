from numpy import genfromtxt
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import KFold


class CustomTool():
    def read_from_abalone(self):
        toDelete = 0
        datapath = '../benchmark_datasets/Abalone/abalone.data'
        my_data = genfromtxt(datapath, delimiter=',')
        new_data = np.delete(my_data, toDelete, 1)
        arr_length = len(new_data[0, :])
        # print(arr_length)
        attr = new_data[:, 0:arr_length-1]
        labels = new_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return self.normalize_attr(attr), self.relabel_data(labels[:, 0], 10)

    def read_from_bank(self):
        datapath = '../benchmark_datasets/Bank/Bank32nh/bank32nh.data'
        my_data = genfromtxt(datapath, delimiter=' ')
        arr_length = len(my_data[0, :])
        # print(arr_length)
        attr = my_data[:, 0:arr_length-1]
        labels = my_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return self.normalize_attr(attr), self.relabel_data(labels[:, 0], 10)

    def read_from_file(self, datapath):
        my_data = genfromtxt(datapath, delimiter=',')
        arr_length = len(my_data[0, :])
        # print(arr_length)
        attr = my_data[:, 0:arr_length-1]
        labels = my_data[:, arr_length-1:arr_length]
        # print(attr)
        # print(labels)
        return self.normalize_attr(attr), self.relabel_data(labels[:, 0], 10)

    def read_from_medical_data(self, datapath):
        my_data = genfromtxt(datapath, delimiter=',', skip_header=1)
        arr_length = len(my_data[0, :])
        # print(arr_length)
        attr = my_data[:, 1:arr_length]
        labels = my_data[:, 0] - 1
        # print(attr)
        # print(labels)
        return self.normalize_attr(attr), labels

    def normalize_attr(self, toy_data):
        rows, dimen = toy_data.shape

        for i in range(dimen):
            mean = toy_data[:, i].mean()
            std = toy_data[:, i].std()
            toy_data[:, i] = (toy_data[:, i] - mean) / std

        return toy_data

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

    def up_sample(self, x, y):
        init_shape = x.shape
        unique, counts = np.unique(y, return_counts=True)
        max_count = int(counts.max())
        new_x = np.array([])
        new_y = np.array([])

        for cls in unique:
            subset_x = np.array(x[y == cls])
            subset_y = np.ones([max_count, 1]) * cls
            subset_x = resample(subset_x, n_samples=max_count, random_state=0)
            new_x = np.append(new_x, subset_x)
            new_y = np.append(new_y, subset_y)

        new_x = new_x.reshape(new_x.size // init_shape[1], init_shape[1])
        return new_x, new_y

    # 5-fold cross validation in default
    def cross_validation(self, data, labels, fold=5):
        length = len(data)
        if length < fold:
            raise ValueError(
                " fold number {} is larger than data length {}".format(fold, length))
        kf = KFold(n_splits=fold, random_state=None, shuffle=True)

        train_list = []
        test_list = []
        for train_index, test_index in kf.split(data):
            X_train, X_test = data[train_index], data[test_index]
            Y_train, Y_test = labels[train_index], labels[test_index]
            train_list.append([X_train, Y_train])
            test_list.append([X_test, Y_test])

        return train_list, test_list

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

    def get_iteration(self, gtol, initial_lr, final_lr, max_iter=2500):
        return min(int(initial_lr / (final_lr * gtol) + 1 - 1 / gtol), max_iter)

