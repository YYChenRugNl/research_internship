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
save_path = '../p_results/p_results.csv'

datapath = 'C:/Users/Yukki/Desktop/RIntern/data_ordinal.csv'
tools = CustomTool()
real_data, real_label = tools.read_from_medical_data(datapath)

cross_validation = 10
real_data, real_label = tools.up_sample(real_data, real_label)
train_list, test_list = tools.cross_validation(real_data, real_label, cross_validation)

gtol_list = [0.05, 0.02, 0.01, 0.005]
number_prototype_list = [1, 2, 3, 4, 5]
kernel_size = [0, 1, 2]
sigma1_list = [0.2, 0.5, 1]
sigma2_list = [0.2, 0.5, 1]
sigma3_list = [0.2, 0.5, 1]

lr_prototype = 0.1
lr_omega = 0.05
final_lr = 0.01

max_iteration = tools.get_iteration(gtol=0.01, initial_lr=lr_prototype, final_lr=final_lr)
max_iteration = 10

# p version
start = time.time()
for number_prototype in number_prototype_list:
    for k in kernel_size:
        for sigma in sigma1_list:
            ab_accuracy = 0
            MAE_sum = 0
            for idx in range(len(train_list)):
                train_data = train_list[idx][0]
                train_label = train_list[idx][1]
                test_data = test_list[idx][0]
                test_label = test_list[idx][1]

                # gmlvq = GmlvqModel(number_prototype)
                # gmlvq.fit(train_data, train_label)
                # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
                # accuracy += gmlvq.score(test_data, test_label)

                ogmlvq = OGmlvqModel(number_prototype, kernel_size=k, gtol=0.005, lr_prototype=0.1, lr_omega=0.05, final_lr=0.01,
                                     batch_flag=False, max_iter=max_iteration, sigma=sigma)
                ogmlvq.fit(train_data, train_label)
                # plot2d(ogmlvq, test_data, test_label, 1, 'ogmlvq', no_index=True)
                score, ab_score, MAE, max_iters = ogmlvq.score(test_data, test_label)
                ab_accuracy += ab_score
                MAE_sum += MAE
                # print('ogmlvq classification accuracy:', score)
                # print('ogmlvq classification ab_accuracy:', ab_score)
                # print('ogmlvq classification MAE:', MAE)

                # aogmlvq = AOGmlvqModel(1, kernel_size=1, gtol=0.05, lr_prototype=0.1, lr_omega=0.05, final_lr=0.01, sigma3=0.5)
                # aogmlvq.fit(train_data, train_label)
                # plot2d(aogmlvq, test_data, test_label, 1, 'aogmlvq', no_index=True)
                # score, ab_score, MAE = aogmlvq.score(test_data, test_label)
                # ab_accuracy += ab_score
                # MAE_sum += MAE
                # print('aogmlvq classification accuracy:', score)
                # print('aogmlvq classification ab_accuracy:', ab_score)
                # print('aogmlvq classification MAE:', MAE)

            average_accuracy = ab_accuracy/len(train_list)
            average_MAE = MAE_sum/len(train_list)
            print('aogmlvq classification average accuracy:', average_accuracy)
            print('aogmlvq classification average MAE:', average_MAE)
            print('number of prototypes:', number_prototype)

            # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'MZE', 'MAE'
            df.loc[df.shape[0]] = np.array([max_iters, number_prototype, k, sigma, 1-average_accuracy, average_MAE])
            df.to_csv(save_path)

end = time.time()
print(end - start, "s")
plt.show()

