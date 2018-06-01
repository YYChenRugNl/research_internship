import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

from glvq import AOGmlvqModel, OGmlvqModel, plot2d, CustomTool, GmlvqModel

import cProfile
import re


print(__doc__)


def test():
    # datapath = '../benchmark_datasets/Machine-Cpu/machine.data'
    datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
    tools = CustomTool()
    # toy_data, toy_label = tools.read_from_abalone()
    # toy_data, toy_label = tools.read_from_bank()
    toy_data, toy_label = tools.read_from_medical_data(datapath)
    # toy_label[toy_label > 0.666] = 2
    # toy_label[(toy_label > 0.333) & (toy_label <= 0.666)] = 1
    # toy_label[toy_label <= 0.333] = 0

    # for i in range(4):
    #     percent = i * 25
    #     fraction = np.percentile(toy_label, percent)
    #     print(fraction)

    # toy_label[toy_label > 0.5] = 1
    # toy_label[(toy_label > 0.1509558) & (toy_label <= 0.2602056)] = 6
    # toy_label[(toy_label > 0.0862336) & (toy_label <= 0.1509558)] = 5
    # toy_label[(toy_label > 0.0484986) & (toy_label <= 0.0862336)] = 4
    # toy_label[(toy_label > 0.025383) & (toy_label <= 0.0484986)] = 3
    # toy_label[(toy_label > 0.0254) & (toy_label <= 0.1126)] = 2
    # toy_label[(toy_label > 0.0019) & (toy_label <= 0.0254)] = 1
    # toy_label[toy_label <= 0.5] = 0


    basic_matrix = [[0.1, 0], [0, 0.1]]
    y_matrix = [[1, 0], [0, 3]]
    x_matrix = [[5, 0], [0, 1]]

    # list_center = [[0, 0], [4, 0], [4, 4], [0, 4], [10, 10], [14, 10], [14, 14], [10, 14]]
    # list_label = [0, 1, 2, 3, 0, 1, 2, 3]
    # list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

    # list_center = [[0, 0], [4, 0], [8, 0], [8, 4], [4, 4], [0, 4]]
    # list_label = [0, 1, 2, 3, 4, 5]
    # list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

    list_center = [[0, 0], [4, 0], [8, 0], [12, 0], [12, 4], [8, 4], [4, 4], [0, 4]]
    list_label = [0, 1, 2, 3, 4, 5, 6, 7]
    list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

    # list_center = [[0, 0], [4, 0], [4, 4], [0, 4]]
    # list_label = [0, 1, 2, 3]
    # list_matrix = [y_matrix, basic_matrix, basic_matrix, y_matrix]

    number_sample = 50
    normalize_flag = True
    # toy_data, toy_label = tools.artificial_data(number_sample, list_center, list_label, list_matrix, normalize_flag)
    toy_data, toy_label = tools.up_sample(toy_data, toy_label)
    toy_train_list, toy_test_list = tools.cross_validation(toy_data, toy_label, 2)

    # print(toy_data)
    # print(toy_label)

    run_flag = True
    # run_flag = False

    if run_flag:
        start = time.time()
        # for number_prototype in [1, 2, 3, 4, 5, 6, 7]:
        for number_prototype in [2]:
            ab_accuracy = 0
            MAE_sum = 0
            for idx in range(len(toy_train_list)):
                train_data = toy_train_list[idx][0]
                train_label = toy_train_list[idx][1]
                test_data = toy_test_list[idx][0]
                test_label = toy_test_list[idx][1]

                # gmlvq = GmlvqModel(number_prototype)
                # gmlvq.fit(train_data, train_label)
                # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
                # accuracy += gmlvq.score(test_data, test_label)

                ogmlvq = OGmlvqModel(1, kernel_size=1, gtol=0.05, lr_prototype=0.1, lr_omega=0.05, final_lr=0.01, batch_flag=False)
                ogmlvq.fit(train_data, train_label)
                # plot2d(ogmlvq, test_data, test_label, 1, 'ogmlvq', no_index=True)
                score, ab_score, MAE, max_iters = ogmlvq.score(test_data, test_label)
                ab_accuracy += ab_score
                MAE_sum += MAE
                print('ogmlvq classification accuracy:', score)
                print('ogmlvq classification ab_accuracy:', ab_score)
                print('ogmlvq classification MAE:', MAE)

                # aogmlvq = AOGmlvqModel(1, kernel_size=1, gtol=0.05, lr_prototype=0.1, lr_omega=0.05, final_lr=0.01, sigma3=0.5)
                # aogmlvq.fit(train_data, train_label)
                # # plot2d(aogmlvq, test_data, test_label, 1, 'aogmlvq', no_index=True)
                # score, ab_score, MAE = aogmlvq.score(test_data, test_label)
                # ab_accuracy += ab_score
                # MAE_sum += MAE
                # print('aogmlvq classification accuracy:', score)
                # print('aogmlvq classification ab_accuracy:', ab_score)
                # print('aogmlvq classification MAE:', MAE)

            average_accuracy = ab_accuracy/len(toy_train_list)
            average_MAE = MAE_sum/len(toy_train_list)
            print('aogmlvq classification average accuracy:', average_accuracy)
            print('aogmlvq classification average MAE:', average_MAE)
            print('number of prototypes:', number_prototype)
        end = time.time()
        print(end - start, "s")
        plt.show()


cProfile.run('test()')
# test()
