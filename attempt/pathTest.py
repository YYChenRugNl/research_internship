import csv
from numpy import genfromtxt
import numpy as np

datapath = '../benchmark_datasets/Abalone/abalone.data'
datapath = '../benchmark_datasets/Bank/Bank32nh/bank32nh.data'
domainpath = '../benchmark_datasets/Abalone/abalone.domain'
domainpath = '../benchmark_datasets/Bank/Bank32nh/bank32nh.domain'


my_data = genfromtxt(datapath, delimiter=' ')
print(my_data)
# new_data = my_data

# arr_length = len(new_data[0, :])
# print(arr_length)
# attr = new_data[:, 0:arr_length-1]
# labels = new_data[:, arr_length-1:arr_length]
# print(attr)
# print(labels)

# with open(datapath, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in spamreader:
#         print(' '.join(row))
#
#
# with open(domainpath, newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=':', quotechar='|')
#     for row in spamreader:
#         print(row)
#         print(row[1])


