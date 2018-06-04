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
            'MZE': [],
            'MAE': []}
df = pd.DataFrame(raw_data, columns=['iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'])
save_path = '../a_results/a_results_' + str(int(time.time())) + '.csv'
tools = CustomTool()

# # datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
datapath = '../benchmark_datasets/Machine-Cpu/machine.data'
#
# # real_data, real_label = tools.read_from_medical_data(datapath)
real_data, real_label = tools.read_from_file(datapath)
#
cross_validation = 10
real_data, real_label = tools.up_sample(real_data, real_label)
train_list, test_list = tools.cross_validation(real_data, real_label, cross_validation)


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
number_prototype_list = [1, 3, 5]
kernel_size = [0, 1, 2]
# sigma1_list = [0.2, 0.5, 1]
sigma1_list = [0.5]
sigma2_list = [0.2, 0.5, 1]
# sigma3_list = [0.2, 0.5, 1]

lr_prototype = 0.1
lr_omega = 0.05
final_lr = 0.01
iteration = 50

gtol = tools.set_iteration(iter=iteration, initial_lr=lr_prototype, final_lr=final_lr)
print('gtol', gtol)

# a version
start = time.time()
for number_prototype in number_prototype_list:
    for k in kernel_size:
        for s1 in sigma1_list:
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
                                       sigma1=s1, sigma2=s1, sigma3=1, n_interval=5)
                aogmlvq, epoch_MZE_MAE_dic = aogmlvq.fit(train_data, train_label, test_data, test_label)
                MZE_MAE_dic_list.append(epoch_MZE_MAE_dic)

            key_list = MZE_MAE_dic_list[0].keys()
            for key in key_list:
                average_MZE_MAE = sum(np.array(dic[key]) for dic in MZE_MAE_dic_list) / len(MZE_MAE_dic_list)
                average_MZE = average_MZE_MAE[0]
                average_MAE = average_MZE_MAE[1]

                # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
                df.loc[df.shape[0]] = np.array([key, number_prototype, k, s1, average_MZE, average_MAE])

                print('aogmlvq classification Epoch:', key)
                print('aogmlvq classification average MZE:', average_MZE)
                print('aogmlvq classification average MAE:', average_MAE)
                print('number of prototypes:', number_prototype)
                print('tolerance:', k)

            df.to_csv(save_path)

end = time.time()
print(end - start, "s")
plt.show()
