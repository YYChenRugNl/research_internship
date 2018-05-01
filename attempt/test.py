import matplotlib.pyplot as plt
import numpy as np
import csv

from glvq import GmlvqModel, plot2d, tools

print(__doc__)

datapath = '../benchmark_datasets/Abalone/abalone.data'
datapath = '../benchmark_datasets/Bank/Bank32nh/bank32nh.data'
domainpath = '../benchmark_datasets/Abalone/abalone.domain'
tools = tools.CustomTool()
# toy_data, toy_label = tools.read_from_file(datapath, 0)
toy_data, toy_label = tools.read_from_file2(datapath)
# toy_label[toy_label > 0.666] = 2
# toy_label[(toy_label > 0.333) & (toy_label <= 0.666)] = 1
# toy_label[toy_label <= 0.333] = 0

for i in range(4):
    percent = i * 25
    fraction = np.percentile(toy_label, percent)
    print(fraction)

toy_label[toy_label > 0.5] = 1
# toy_label[(toy_label > 0.1509558) & (toy_label <= 0.2602056)] = 6
# toy_label[(toy_label > 0.0862336) & (toy_label <= 0.1509558)] = 5
# toy_label[(toy_label > 0.0484986) & (toy_label <= 0.0862336)] = 4
# toy_label[(toy_label > 0.025383) & (toy_label <= 0.0484986)] = 3
# toy_label[(toy_label > 0.0254) & (toy_label <= 0.1126)] = 2
# toy_label[(toy_label > 0.0019) & (toy_label <= 0.0254)] = 1
toy_label[toy_label <= 0.5] = 0

print(toy_data)
print(toy_label)

run_flag = True
# run_flag = False

print('GMLVQ:')

if run_flag:
    gmlvq = GmlvqModel()
    gmlvq.fit(toy_data, toy_label)
    plot2d(gmlvq, toy_data, toy_label, 1, 'gmlvq')
    print('classification accuracy:', gmlvq.score(toy_data, toy_label))
    plt.show()
