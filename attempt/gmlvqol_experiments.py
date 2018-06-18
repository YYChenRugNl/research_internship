import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

from glvq import GmlvqOLModel, AOGmlvqModel, OGmlvqModel, simple_line_plot, plot2d, CustomTool, GmlvqModel

raw_data = {'epoch': [],
            'nb_prototypes': [],
            'MZE': [],
            'MAE': []}
df = pd.DataFrame(raw_data, columns=['epoch', 'nb_prototypes', 'MZE', 'MAE'])
save_path = '../gmlvq_results/gmlvqol_results_' + str(int(time.time())) + '.csv'

datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
# datapath = '../benchmark_datasets/Machine-Cpu/machine.data'
tools = CustomTool()
real_data, real_label = tools.read_from_medical_data(datapath)
# real_data, real_label = tools.read_from_file(datapath)

cross_validation = 8
# real_data, real_label = tools.up_sample(real_data, real_label)
train_list, test_list = tools.cross_validation_by_class(real_data, real_label, cross_validation)

# gtol_list = [0.05, 0.02, 0.01, 0.005]
number_prototype_list = [5]

final_run = True
times = 5

if final_run:
    # parameters
    number_prototype = 5
    iter_number = 50
    initial_lr = 0.2
    final_lr = 0.1
    zeropoint = 0.8
    sigma = 0.5
    sigma1 = 1
    sigma2 = 0.2
    kernel_size = 1
    n_interval = 5
    early_stop = 185


    all_fold_cost = []
    TOTAL_LIST = []
    ab_accuracy_sum = 0
    MAE_sum = 0

    MZE_final_sum = 0
    MAE_final_sum = 0
    for n_run in range(times):
        start = time.time()

        train_list, test_list = tools.cross_validation_by_class(real_data, real_label, cross_validation)
        gtol = tools.set_iteration(iter=iter_number, initial_lr=initial_lr, final_lr=final_lr)
        print('gtol', gtol)

        MAE_sum = 0
        MZE_sum = 0
        MZE_MAE_dic_list = []
        for idx in range(len(train_list)):
            train_data = train_list[idx][0]
            train_label = train_list[idx][1]
            test_data = test_list[idx][0]
            test_label = test_list[idx][1]

            gmlvqol = GmlvqOLModel(number_prototype, kernel_size=0, gtol=gtol, lr_prototype=initial_lr,
                                   lr_omega=initial_lr * 0.8,
                                   final_lr=final_lr, batch_flag=False, n_interval=20, max_iter=iter_number)
            gmlvqol, epoch_MZE_MAE_dic, proto_history_list = gmlvqol.fit(train_data, train_label, test_data,
                                                                         test_label, trace_proto=True)
            # ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=False)
            # plot2d(gmlvqol, test_data, test_label, proto_history_list, figure=1, prototype_count=number_prototype, title='online_gmlvq', no_index=True)
            MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)

        TOTAL_LIST.append(MZE_MAE_dic_list)
        key_list = list(MZE_MAE_dic_list[0].keys())
        key_length = len(key_list)
        for key_index in range(key_length):
            key = key_list[key_index]
            average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
            average_MZE = average_MZE_MAE[0]
            average_MAE = average_MZE_MAE[1]

            # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
            df.loc[df.shape[0]] = np.array([key, number_prototype, average_MZE, average_MAE])
            print('aogmlvq classification Epoch:', key)
            print('aogmlvq classification average MZE:', average_MZE)
            print('aogmlvq classification average MAE:', average_MAE)
            print('number of prototypes:', number_prototype)
            if key_index == key_length - 1:
                MZE_final_sum += average_MZE
                MAE_final_sum += average_MAE

        df.to_csv(save_path)

        end = time.time()
        print(end - start, "s")

    mze_list = []
    mae_list = []
    for MZE_MAE_dic_list in TOTAL_LIST:
        avg_MZE_MAE = sum(np.array(list(dic.values())) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
        avg_MZE = avg_MZE_MAE[:, 0]
        avg_MAE = avg_MZE_MAE[:, 1]
        mze_list.append(avg_MZE)
        mae_list.append(avg_MAE)

    # final_MZE = MZE_final_sum / times
    # final_MAE = MAE_final_sum / times
    final_MZE_epoch = sum(np.array(mze_list)) / len(mze_list)
    final_MAE_epoch = sum(np.array(mae_list)) / len(mae_list)
    print("final MZE:", final_MZE_epoch)
    print("final MAE:", final_MAE_epoch)
    simple_line_plot(key_list, final_MAE_epoch, 'MAE of each epoch', 1, 1, 1)
    for df_idx in range(len(final_MAE_epoch)):
        epoch = key_list[df_idx]
        final_MZE = final_MZE_epoch[df_idx]
        final_MAE = final_MAE_epoch[df_idx]
        df.loc[df.shape[0]] = np.array(['final', number_prototype, final_MZE, final_MAE])
    df.to_csv(save_path)

else:
    start = time.time()
    for number_prototype in number_prototype_list:
        MZE_MAE_dic_list = []
        accuracy = 0
        MAE_SUM = 0
        for idx in range(len(train_list)):
            train_data = train_list[idx][0]
            train_label = train_list[idx][1]
            test_data = test_list[idx][0]
            test_label = test_list[idx][1]

            gmlvqol = GmlvqOLModel(number_prototype, kernel_size=0, gtol=gtol, lr_prototype=initial_lr,
                                   lr_omega=initial_lr * 0.8,
                                   final_lr=final_lr, batch_flag=False, n_interval=20, max_iter=iter_number)
            gmlvqol, epoch_MZE_MAE_dic, proto_history_list = gmlvqol.fit(train_data, train_label, test_data,
                                                                         test_label, trace_proto=True)
            # ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label, trace_proto=False)
            # plot2d(gmlvqol, test_data, test_label, proto_history_list, figure=1, prototype_count=number_prototype, title='online_gmlvq', no_index=True)
            MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)

        print('average_accuracy:', 1 - accuracy/len(test_list))
        print('average_MAE:', MAE_SUM/len(test_list))

                #     ogmlvq = OGmlvqModel(number_prototype, kernel_size=k, gtol=gtol, lr_prototype=lr_prototype, lr_omega=lr_omega, final_lr=final_lr,
                #                          batch_flag=False, sigma=sigma, n_interval=5)
                #     ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label)
                #     MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)
                #     # plot2d(ogmlvq, test_data, test_label, 1, 'ogmlvq', no_index=True)
                #     # print('ogmlvq classification accuracy:', score)
                #     # print('ogmlvq classification ab_accuracy:', ab_score)
                #     # print('ogmlvq classification MAE:', MAE)
                #
                # key_list = MZE_MAE_dic_list[0].keys()
                # for key in key_list:
                #     average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
                #     average_MZE = average_MZE_MAE[0]
                #     average_MAE = average_MZE_MAE[1]
                #     # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
                #     df.loc[df.shape[0]] = np.array([key, number_prototype, k, sigma, average_MZE, average_MAE])
                #
                    # print('aogmlvq classification Epoch:', key)
                    # print('aogmlvq classification average MZE:', average_MZE)
                    # print('aogmlvq classification average MAE:', average_MAE)
                    # print('number of prototypes:', number_prototype)
                #
                # df.to_csv(save_path)

    end = time.time()
    print(end - start, "s")
    plt.show()

