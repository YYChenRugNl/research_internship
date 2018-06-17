import pandas as pd
import h5py
import numexpr

filename1 = 'Magphys_GZ_bulge_dominance_and_spiralwindings_20h42_14June2018.h5'
filename2 = 'GFS_GZ_bulge_dominance_and_spiralwindings_20h41_14June2018.h5'

with pd.HDFStore(filename1) as store:
    df = store['df']


f = h5py.File(filename1, 'r')

# List all groups
print("Keys: %s" % f.keys())


a_group_key = list(f.keys())[0]

# Get the data
data = list(f[a_group_key])
print(data)