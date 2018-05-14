import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv

from glvq import OGmlvqModel, plot2d, CustomTool, GmlvqModel

print(__doc__)

# datapath = '../benchmark_datasets/Abalone/abalone.data'
# datapath = '../benchmark_datasets/Bank/Bank32nh/bank32nh.data'
# domainpath = '../benchmark_datasets/Abalone/abalone.domain'
tools = CustomTool()
# toy_data, toy_label = tools.read_from_file(datapath, 0)
# toy_data, toy_label = tools.read_from_file2(datapath)
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

# list_center = [[0, 0], [4, 0], [4, 4], [0, 4], [10, 10], [14, 10], [14, 14], [10, 14]]
# list_label = [0, 1, 2, 3, 0, 1, 2, 3]
# list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

list_center = [[0, 0], [4, 0], [8, 0], [8, 4], [4, 4], [0, 4]]
list_label = [0, 1, 2, 3, 4, 5]
list_matrix = [basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix, basic_matrix]

number_sample = 10
normalize_flag = True
toy_data, toy_label = tools.artificial_data(number_sample, list_center, list_label, list_matrix, normalize_flag)

# print(toy_data)
# print(toy_label)

run_flag = True
# run_flag = False

print('OGMLVQ:')

if run_flag:
    gmlvq = GmlvqModel(1)
    gmlvq.fit(toy_data, toy_label)
    plot2d(gmlvq, toy_data, toy_label, 1, 'gmlvq')
    print('classification accuracy:', gmlvq.score(toy_data, toy_label))

    ogmlvq = OGmlvqModel(1, gtol=0.001)
    ogmlvq.fit(toy_data, toy_label)
    plot2d(ogmlvq, toy_data, toy_label, 1, 'ogmlvq')
    print('classification accuracy:', ogmlvq.score(toy_data, toy_label))

    plt.show()
