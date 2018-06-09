import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

from glvq import AOGmlvqModel, OGmlvqModel, plot2d, CustomTool, GmlvqModel

raw_data = {'iterations': [],
            'nb_prototypes': [],
            'k_size': [],
            'sigma': [],
            'sigma1': [],
            'MZE': [],
            'MAE': [],
            'lr_prototype': [],
            'lr_omega': [],
            'lr_final': [],
            'max-iteration': []}

df = pd.DataFrame(raw_data, columns=['iterations', 'nb_prototypes', 'k_size', 'sigma', 'sigma1', 'MZE', 'MAE',
                                     'lr_prototype', 'lr_omega', 'lr_final', 'max-iteration'])
save_path = '../p_results/p_results_' + str(int(time.time())) + '.csv'

datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
# datapath = '../benchmark_datasets/Machine-Cpu/machine.data'
tools = CustomTool()
real_data, real_label = tools.read_from_medical_data(datapath)
# real_data, real_label = tools.read_from_file(datapath)

# basic_matrix = [[0.1, 0], [0, 0.1]]
# list_center = [[0, 0], [4, 0], [8, 0], [12, 0], [12, 4], [8, 4], [4, 4], [0, 4]]
# list_label = [0, 1, 2, 3, 4, 5, 6, 7]
# list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]
# number_sample = 50
# normalize_flag = True
# real_data, real_label = tools.artificial_data(number_sample, list_center, list_label, list_matrix, normalize_flag)

cross_validation = 10
real_data, real_label = tools.up_sample(real_data, real_label)
train_list, test_list = tools.cross_validation(real_data, real_label, cross_validation)

# gtol_list = [0.05, 0.02, 0.01, 0.005]
number_prototype_list = [5]
kernel_size = [0]
sigma_list = [0.5]
sigma1_list = [0.7]

# lr_prototype = 0.1
# lr_omega = 0.08
# final_lr = 0.01
max_iteration = 800
# lr_prototype_list = [0.05, 0.1, 0.2]
# lr_omega_list = [0.025, 0.08, 0.1]
# final_lr_list = [0.005, 0.01, 0.02]

lr_prototype_list = [0.1]
lr_omega_list = [0.08]
final_lr_list = [0.01]

final_run = True
times = 30

if final_run:
    # parameters
    number_prototype = 5
    k = 0
    sigma = 0.5
    sigma1 = 0.7
    lr_prototype = 0.1
    lr_omega = 0.08
    final_lr = 0.01

    MZE_final_sum = 0
    MAE_final_sum = 0
    for n_run in range(times):
        # p version
        start = time.time()

        gtol = tools.set_iteration(iter=max_iteration, initial_lr=lr_prototype, final_lr=final_lr)
        print('gtol', gtol)

        MZE_MAE_dic_list = []
        for idx in range(len(train_list)):
            train_data = train_list[idx][0]
            train_label = train_list[idx][1]
            test_data = test_list[idx][0]
            test_label = test_list[idx][1]

            ogmlvq = OGmlvqModel(number_prototype, kernel_size=k, gtol=gtol,
                                 lr_prototype=lr_prototype, lr_omega=lr_omega, final_lr=final_lr,
                                 batch_flag=False, sigma=sigma, n_interval=10, sigma1=sigma1)
            ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label)
            MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)
            # plot2d(ogmlvq, test_data, test_label, 1, 'ogmlvq', no_index=True)
            # print('ogmlvq classification accuracy:', score)
            # print('ogmlvq classification ab_accuracy:', ab_score)
            # print('ogmlvq classification MAE:', MAE)

        key_list = list(MZE_MAE_dic_list[0].keys())
        key_length = len(key_list)
        for key_index in range(key_length):
            key = key_list[key_index]
            average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(
                MZE_MAE_dic_list)
            average_MZE = average_MZE_MAE[0]
            average_MAE = average_MZE_MAE[1]
            # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
            df.loc[df.shape[0]] = np.array(
                [key, number_prototype, k, sigma, sigma1, average_MZE, average_MAE,
                 lr_prototype, lr_omega, final_lr, max_iteration])

            print('ogmlvq classification Epoch:', key)
            print('ogmlvq classification average MZE:', average_MZE)
            print('ogmlvq classification average MAE:', average_MAE)
            print('number of prototypes:', number_prototype)
            if key_index == key_length - 1:
                MZE_final_sum += average_MZE
                MAE_final_sum += average_MAE

        df.to_csv(save_path)

        end = time.time()
        print(end - start, "s")
        plt.show()

    final_MZE = MZE_final_sum/times
    final_MAE = MAE_final_sum/times
    df.loc[df.shape[0]] = np.array(
        [00, number_prototype, k, sigma, sigma1, final_MZE, final_MAE,
         lr_prototype, lr_omega, final_lr, max_iteration])
    df.to_csv(save_path)
    print("final MAE:", final_MAE)

else:

    # p version
    start = time.time()
    for number_prototype in number_prototype_list:
        for k in kernel_size:
            for sigma in sigma_list:
                for sigma1 in sigma1_list:
                    for index in range(len(lr_prototype_list)):
                        lr_prototype = lr_prototype_list[index]
                        lr_omega = lr_omega_list[index]
                        final_lr = final_lr_list[index]
                        gtol = tools.set_iteration(iter=max_iteration, initial_lr=lr_prototype, final_lr=final_lr)
                        print('gtol', gtol)

                        MZE_MAE_dic_list = []
                        for idx in range(len(train_list)):
                            train_data = train_list[idx][0]
                            train_label = train_list[idx][1]
                            test_data = test_list[idx][0]
                            test_label = test_list[idx][1]

                            ogmlvq = OGmlvqModel(number_prototype, kernel_size=k, gtol=gtol,
                                                 lr_prototype=lr_prototype, lr_omega=lr_omega, final_lr=final_lr,
                                                 batch_flag=False, sigma=sigma, n_interval=10, sigma1=sigma1)
                            ogmlvq, epoch_MZE_MAE_dic = ogmlvq.fit(train_data, train_label, test_data, test_label)
                            MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)
                            # plot2d(ogmlvq, test_data, test_label, 1, 'ogmlvq', no_index=True)
                            # print('ogmlvq classification accuracy:', score)
                            # print('ogmlvq classification ab_accuracy:', ab_score)
                            # print('ogmlvq classification MAE:', MAE)

                        key_list = MZE_MAE_dic_list[0].keys()
                        for key in key_list:
                            average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
                            average_MZE = average_MZE_MAE[0]
                            average_MAE = average_MZE_MAE[1]
                            # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
                            df.loc[df.shape[0]] = np.array([key, number_prototype, k, sigma, sigma1, average_MZE, average_MAE,
                                                            lr_prototype, lr_omega, final_lr, max_iteration])

                            print('ogmlvq classification Epoch:', key)
                            print('ogmlvq classification average MZE:', average_MZE)
                            print('ogmlvq classification average MAE:', average_MAE)
                            print('number of prototypes:', number_prototype)

                        df.to_csv(save_path)

    end = time.time()
    print(end - start, "s")
    plt.show()

