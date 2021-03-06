import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

from glvq import AOGmlvqModel, OGmlvqModel, simple_line_plot, plot2d, CustomTool, GmlvqModel

raw_data = {'iterations': [],
            'nb_prototypes': [],
            'k_size': [],
            'sigma1': [],
            'sigma2': [],
            'sigma3': [],
            'MZE': [],
            'MAE': [],
            'lr_prototype': [],
            'lr_omega': [],
            'lr_final': [],
            'iteration': [],
            'zeropoint': []}
df = pd.DataFrame(raw_data, columns=['iterations', 'nb_prototypes', 'k_size', 'sigma1', 'sigma2', 'sigma3', 'MZE', 'MAE',
                                     'lr_prototype', 'lr_omega', 'lr_final', 'iteration', 'zeropoint'])
save_path = '../a_results/a_results_' + str(int(time.time())) + '.csv'
tools = CustomTool()

datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
# datapath = '../benchmark_datasets/Machine-Cpu/machine.data'
#
real_data, real_label = tools.read_from_medical_data(datapath)
# real_data, real_label = tools.read_from_file(datapath)
#
cross_validation = 8
# real_data, real_label = tools.up_sample(real_data, real_label)
train_list, test_list = tools.cross_validation_by_class(real_data, real_label, cross_validation)


##########
basic_matrix = [[0.1, 0], [0, 0.1]]

# list_center = [[0, 0], [4, 0], [4, 4], [0, 4], [10, 10], [14, 10], [14, 14], [10, 14]]
# list_label = [0, 1, 2, 3, 0, 1, 2, 3]
# list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

# list_center = [[0, 0], [4, 0], [8, 0], [8, 4], [4, 4], [0, 4]]
# list_label = [0, 1, 2, 3, 4, 5]
# list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

list_center = [[0, 0], [4, 0], [8, 0], [12, 0], [12, 4], [8, 4], [4, 4], [0, 4]]
list_label = [0, 1, 2, 3, 4, 5, 6, 7]
list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

number_sample = 50
normalize_flag = True
# toy_data, toy_label = tools.artificial_data(number_sample, list_center, list_label, list_matrix, normalize_flag)
# train_list, test_list = tools.cross_validation(toy_data, toy_label, 4)
##########

# gtol_list = [0.05, 0.02, 0.01, 0.005]
# number_prototype_list = [1, 2, 3, 4, 5]
number_prototype_list = [5]
kernel_size = [0, 1]
# sigma1_list = [0.2, 0.5, 1]
sigma1_list = [0.2, 0.4]
# sigma2_list = [0.3]
# sigma3_list = [0.2, 0.5, 1]


lr_prototype_list = [0.05, 0.07, 0.1, 0.12]
lr_omega_list = [0.025, 0.05, 0.08, 0.1]
final_lr_list = [0.005, 0.007, 0.001, 0.012]
iteration_list = [200, 600, 1000]
zeropoint_list = [0.9, 0.95]
# lr_prototype_list = [0.1]
# lr_omega_list = [0.08]
# final_lr_list = [0.01]
iteration_list = [200]


final_run = True
times = 30

if final_run:
    # parameters
    number_prototype = 5
    k = 1
    sigma1 = 1
    sigma2 = 0.2
    sigma3 = 1
    lr_prototype = 0.005
    lr_omega = lr_prototype * 0.8
    final_lr = 0.001
    # final_lr = 0.08
    max_iteration = 500
    zeropoint = 1
    n_interval = 1
    early_stop = 500

    MZE_final_sum = 0
    MAE_final_sum = 0
    TOTAL_LIST = []
    key_list = []
    for n_run in range(times):
        start = time.time()
        gtol = tools.set_iteration(iter=max_iteration, initial_lr=lr_prototype, final_lr=final_lr)
        print('gtol', gtol)

        train_list, test_list = tools.cross_validation_by_class(real_data, real_label, cross_validation)

        MZE_MAE_dic_list = []
        for idx in range(len(train_list)):
            train_data = train_list[idx][0]
            train_label = train_list[idx][1]
            test_data = test_list[idx][0]
            test_label = test_list[idx][1]

            # gmlvq = GmlvqModel(number_prototype)
            # gmlvq.fit(train_data, train_label)
            # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
            # accuracy += gmlvq.score(test_data, test_label)

            aogmlvq = AOGmlvqModel(number_prototype, kernel_size=k, gtol=gtol, lr_prototype=lr_prototype,
                                   lr_omega=lr_omega, final_lr=final_lr, max_iter=early_stop,
                                   sigma1=1, sigma2=sigma2, sigma3=1, n_interval=n_interval, zeropoint=zeropoint)
            aogmlvq, epoch_MZE_MAE_dic = aogmlvq.fit(train_data, train_label, test_data, test_label)
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
            df.loc[df.shape[0]] = np.array([key, number_prototype, k, sigma1, sigma2, sigma3, average_MZE, average_MAE,
                                            lr_prototype, lr_omega, final_lr, max_iteration, zeropoint])
            print('aogmlvq classification Epoch:', key)
            print('aogmlvq classification average MZE:', average_MZE)
            print('aogmlvq classification average MAE:', average_MAE)
            print('number of prototypes:', number_prototype)
            print('tolerance:', k)
            if key_index == key_length - 1:
                MZE_final_sum += average_MZE
                MAE_final_sum += average_MAE

        df.to_csv(save_path)

        end = time.time()
        print(end - start, "s")
        plt.show()

    mze_list = []
    mae_list = []
    for MZE_MAE_dic_list in TOTAL_LIST:
        avg_MZE_MAE = sum(np.array(list(dic.values())) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
        avg_MZE = avg_MZE_MAE[:, 0]
        avg_MAE = avg_MZE_MAE[:, 1]
        mze_list.append(avg_MZE)
        mae_list.append(avg_MAE)

    final_MZE_epoch = sum(np.array(mze_list))/len(mze_list)
    final_MAE_epoch = sum(np.array(mae_list))/len(mae_list)

    # final_MZE = MZE_final_sum / times
    # final_MAE = MAE_final_sum / times
    print("final MZE:", final_MZE_epoch)
    print("final MAE:", final_MAE_epoch)
    simple_line_plot(key_list, final_MAE_epoch, 'MAE of each epoch', 1, 1, 1)
    for df_idx in range(len(final_MAE_epoch)):
        epoch = key_list[df_idx]
        final_MZE = final_MZE_epoch[df_idx]
        final_MAE = final_MAE_epoch[df_idx]
        df.loc[df.shape[0]] = np.array([epoch,
            number_prototype, k, sigma1, sigma2, sigma3, final_MZE, final_MAE,
            lr_prototype, lr_omega, final_lr, max_iteration, zeropoint])
    df.to_csv(save_path)


else:
    # a version
    start = time.time()
    for number_prototype in number_prototype_list:
        for k in kernel_size:
            for s1 in sigma1_list:
                for it in iteration_list:
                    for zeropoint in zeropoint_list:
                        for idx in range(len(lr_prototype_list)):
                            sigma1 = 1
                            sigma2 = s1
                            sigma3 = 1
                            lr_prototype = lr_prototype_list[idx]
                            lr_omega = lr_omega_list[idx]
                            final_lr = final_lr_list[idx]
                            gtol = tools.set_iteration(iter=it, initial_lr=lr_prototype, final_lr=final_lr)
                            print('gtol', gtol)

                            MZE_MAE_dic_list = []
                            for idx in range(len(train_list)):
                                train_data = train_list[idx][0]
                                train_label = train_list[idx][1]
                                test_data = test_list[idx][0]
                                test_label = test_list[idx][1]

                                # gmlvq = GmlvqModel(number_prototype)
                                # gmlvq.fit(train_data, train_label)
                                # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
                                # accuracy += gmlvq.score(test_data, test_label)

                                aogmlvq = AOGmlvqModel(number_prototype, kernel_size=k, gtol=gtol, lr_prototype=lr_prototype, lr_omega=lr_omega, final_lr=final_lr,
                                                       sigma1=1, sigma2=s1, sigma3=1, n_interval=10, zeropoint=zeropoint)
                                aogmlvq, epoch_MZE_MAE_dic = aogmlvq.fit(train_data, train_label, test_data, test_label)
                                MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)

                            key_list = MZE_MAE_dic_list[0].keys()
                            for key in key_list:
                                average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
                                average_MZE = average_MZE_MAE[0]
                                average_MAE = average_MZE_MAE[1]

                                # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
                                df.loc[df.shape[0]] = np.array([key, number_prototype, k, sigma1, sigma2, sigma3, average_MZE, average_MAE,
                                                                lr_prototype, lr_omega, final_lr, it, zeropoint])
                                print('aogmlvq classification Epoch:', key)
                                print('aogmlvq classification average MZE:', average_MZE)
                                print('aogmlvq classification average MAE:', average_MAE)
                                print('number of prototypes:', number_prototype)
                                print('tolerance:', k)

                            df.to_csv(save_path)

    end = time.time()
    print(end - start, "s")
    plt.show()

