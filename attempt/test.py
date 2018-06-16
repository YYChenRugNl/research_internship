import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

from glvq import GmlvqOLModel, AOGmlvqModel, OGmlvqModel, simple_line_plot, plot2d, CustomTool, GmlvqModel

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
    # toy_data, toy_label = tools.read_from_file(datapath)

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
    #
    list_center = [[0, 0], [4, 0], [8, 0], [12, 0], [12, 4], [8, 4], [4, 4], [0, 4]]
    list_label = [0, 1, 2, 3, 4, 5, 6, 7]
    list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

    # list_center = [[0, 0], [4, 0], [4, 4], [0, 4]]
    # list_label = [0, 1, 2, 3]
    # list_matrix = [y_matrix, basic_matrix, basic_matrix, y_matrix]

    number_sample = 10
    normalize_flag = True
    # toy_data, toy_label = tools.artificial_data(number_sample, list_center, list_label, list_matrix, normalize_flag)
    # toy_data, toy_label = tools.up_sample(toy_data, toy_label)
    # toy_train_list, toy_test_list = tools.cross_validation(toy_data, toy_label, 8)
    toy_train_list, toy_test_list = tools.cross_validation_by_class(toy_data, toy_label, 8)
    # toy_train_list = [[toy_data, toy_label]]
    # toy_test_list = [[toy_data, toy_label]]

    # print(toy_data)
    # print(toy_label)

    run_gmlvq = False
    run_gmlvqol = False
    run_ogmlvq = False
    run_aogmlvq = False

    # run_gmlvq = True
    # run_gmlvqol = True
    # run_ogmlvq = True
    run_aogmlvq = True

    run_flag = True
    # run_flag = False

    method = ''

    if run_flag:
        start = time.time()
        # for number_prototype in [1, 2, 3, 4, 5, 6, 7]:
        for number_prototype in [5]:
            iter_number = 80
            initial_lr = 0.2
            final_lr = 0.02
            zeropoint = 0.95
            sigma = 0.5
            sigma1 = 1
            sigma2 = 0.2
            kernel_size = 1

            MZE_MAE_dic_list = []
            all_fold_cost = []
            ab_accuracy_sum = 0
            MAE_sum = 0

            print('tolerance:', kernel_size)
            print(initial_lr, final_lr, zeropoint)

            for idx in range(len(toy_train_list)):
            # for idx in range(1):
                train_data = toy_train_list[idx][0]
                train_label = toy_train_list[idx][1]
                test_data = toy_test_list[idx][0]
                test_label = toy_test_list[idx][1]
                gtol = tools.set_iteration(iter=iter_number, initial_lr=initial_lr, final_lr=final_lr)
                print('gtol', gtol)
                if run_gmlvq:
                    method = 'gmlvq'
                    gmlvq = GmlvqModel(number_prototype)
                    gmlvq.fit(train_data, train_label)
                    # plot2d(gmlvq, test_data, test_label, figure=1, prototype_count=number_prototype,
                    #        title='gmlvq', no_index=True)
                    ab_accuracy, MAE = gmlvq.score(test_data, test_label)
                    ab_accuracy_sum += ab_accuracy
                    MAE_sum += MAE
                    # print('gmlvq classification accuracy:', ab_accuracy)
                    # print('gmlvq classification MAE:', MAE)

                if run_gmlvqol:
                    method = 'gmlvq_online'
                    gmlvqol = GmlvqOLModel(number_prototype, kernel_size=0, gtol=0.05, lr_prototype=initial_lr, lr_omega=initial_lr*0.8,
                                         final_lr=final_lr, batch_flag=False, n_interval=20, max_iter=2000)
                    gmlvqol, epoch_MZE_MAE_dic, proto_history_list = gmlvqol.fit(train_data, train_label, test_data,
                                                                               test_label, trace_proto=True)
                    # ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=False)
                    # plot2d(gmlvqol, test_data, test_label, proto_history_list, figure=1, prototype_count=number_prototype, title='online_gmlvq', no_index=True)
                    MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)

                if run_ogmlvq:
                    method = 'ogmlvq'
                    ogmlvq = OGmlvqModel(number_prototype, kernel_size=kernel_size, gtol=gtol, lr_prototype=initial_lr, lr_omega=initial_lr*0.8,
                                         final_lr=final_lr, batch_flag=False, n_interval=10, max_iter=5000, sigma=sigma, sigma1=sigma1, cost_trace=True)
                    ogmlvq, epoch_MZE_MAE_dic, proto_history_list, cost_list = ogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=True)
                    # ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=False)
                    # plot2d(ogmlvq, test_data, test_label, proto_history_list, figure=1, prototype_count=number_prototype, title='p_ogmlvq', no_index=True)
                    MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)
                    all_fold_cost.append(cost_list)
                    key_list = MZE_MAE_dic_list[0].keys()
                    for key in key_list:
                        print(method, 'classification Epoch:', key)
                        print(method, 'classification MZE:', epoch_MZE_MAE_dic[key][0])
                        print(method, 'classification MAE:', epoch_MZE_MAE_dic[key][1])
                    simple_line_plot(list(key_list), cost_list, 'cost of p-OGMLVQ')

                if run_aogmlvq:
                    method = 'aogmlvq'
                    aogmlvq = AOGmlvqModel(number_prototype, kernel_size==kernel_size, gtol=gtol, lr_prototype=initial_lr, lr_omega=initial_lr*0.8,
                                           final_lr=final_lr, sigma3=1, n_interval=1, max_iter=2500, sigma1=1, sigma2=sigma2, cost_trace=True
                                           , zeropoint=zeropoint)
                    aogmlvq, epoch_MZE_MAE_dic, proto_history_list, cost_list = aogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=True)
                    # plot2d(aogmlvq, test_data, test_label,proto_history_list, figure=1, prototype_count=number_prototype,
                    #        title='a_ogmlvq', no_index=True)
                    MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)
                    all_fold_cost.append(cost_list)
                    key_list = MZE_MAE_dic_list[0].keys()
                    for key in key_list:
                        print(method, 'classification Epoch:', key)
                        # print(method, 'classification MZE:', epoch_MZE_MAE_dic[key][0])
                        print(method, 'classification MAE:', epoch_MZE_MAE_dic[key][1])

            if run_gmlvq:
                print('gmlvq classification average MZE:', 1 - ab_accuracy_sum/len(toy_train_list))
                print('gmlvq classification average MAE:', MAE_sum/len(toy_train_list))
            if run_gmlvqol or run_ogmlvq or run_aogmlvq:
                key_list = MZE_MAE_dic_list[0].keys()
                for key in key_list:
                    average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list)/len(MZE_MAE_dic_list)
                    average_MZE = average_MZE_MAE[0]
                    average_MAE = average_MZE_MAE[1]

                    print(method, 'classification Epoch:', key)
                    print(method, 'classification average MZE:', average_MZE)
                    print(method, 'classification average MAE:', average_MAE)
                    print('number of prototypes:', number_prototype)

                avg_cost_list = sum(np.array(each_fold) for each_fold in all_fold_cost)/len(all_fold_cost)
                simple_line_plot(list(key_list), avg_cost_list, 'average cost of a-OGMLVQ')
        end = time.time()
        print(end - start, "s")
        plt.show()


# cProfile.run('test()')
test()
