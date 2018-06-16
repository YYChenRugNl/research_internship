import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time

from glvq import AOGmlvqModel, OGmlvqModel, plot2d, CustomTool, GmlvqModel

raw_data = {'nb_prototypes': [],
            'MZE': [],
            'MAE': []}
df = pd.DataFrame(raw_data, columns=['nb_prototypes', 'MZE', 'MAE'])
save_path = '../gmlvq_results/gmlvq_results_' + str(int(time.time())) + '.csv'

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
times = 20

if final_run:
    # parameters
    number_prototype = 5

    MZE_final_sum = 0
    MAE_final_sum = 0
    for n_run in range(times):
        start = time.time()

        train_list, test_list = tools.cross_validation_by_class(real_data, real_label, cross_validation)

        MAE_sum = 0
        MZE_sum = 0
        for idx in range(len(train_list)):
            train_data = train_list[idx][0]
            train_label = train_list[idx][1]
            test_data = test_list[idx][0]
            test_label = test_list[idx][1]

            # gmlvq = GmlvqModel(number_prototype)
            # gmlvq.fit(train_data, train_label)
            # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
            # accuracy += gmlvq.score(test_data, test_label)

            gmlvq = GmlvqModel(number_prototype)
            gmlvq.fit(train_data, train_label)
            # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
            score, MAE = gmlvq.score(test_data, test_label)
            MAE_sum += MAE
            MZE_sum += 1 - score

        average_MZE = MZE_sum/len(train_list)
        average_MAE = MAE_sum/len(train_list)
        MZE_final_sum += average_MZE
        MAE_final_sum += average_MAE
        print('gmlvq classification average MZE:', average_MZE)
        print('gmlvq classification average MAE:', average_MAE)
        print('number of prototypes:', number_prototype)

        # 'iterations', 'nb_prototypes', 'k_size', 'sigma', 'average_MZE', 'average_MAE'
        df.loc[df.shape[0]] = np.array([number_prototype, average_MZE, average_MAE])
        df.to_csv(save_path)

        end = time.time()
        print(end - start, "s")

    final_MZE = MZE_final_sum / times
    final_MAE = MAE_final_sum / times
    print("final MZE:", final_MZE)
    print("final MAE:", final_MAE)
    df.loc[df.shape[0]] = np.array(['final', final_MZE, final_MAE])
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

            gmlvq = GmlvqModel(number_prototype)
            gmlvq.fit(train_data, train_label)
            # plot2d(gmlvq, test_data, test_label, 1, 'gmlvq')
            score, MAE = gmlvq.score(test_data, test_label)
            print('gmlvq classification MZE:', 1-score)
            print('gmlvq classification MAE:', MAE)
            accuracy += score
            MAE_SUM += MAE
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

