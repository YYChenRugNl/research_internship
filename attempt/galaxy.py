import pandas as pd
# import h5py
# import tables

filename1 = 'Magphys_GZ_bulge_dominance_and_spiralwindings_20h42_14June2018.h5'
filename2 = 'GFS_GZ_bulge_dominance_and_spiralwindings_20h41_14June2018.h5'

with pd.HDFStore(filename2) as store:
    df = store['df']

# df["label"] = " "
print(df.shape)
print(df)
# print(list(df.columns.values))
# print(df.shape)

# df1 = df.loc[df['spiralwinding_loose_frac'] > 0.7]
# df2 = df.loc[df['spiralwinding_medium_frac'] > 0.7]
# df3 = df.loc[df['spiralwinding_tight_frac'] > 0.7]
#
# df4 = df.loc[df['bulge_no_bulge_frac'] > 0.7]
# df5 = df.loc[df['bulge_obvious_frac'] > 0.7]
# df6 = df.loc[df['bulge_dominant_frac'] > 0.7]

# df1 = df.loc[df['spiralwinding_loose_frac'] > 0.9]
# df2 = df.loc[df['spiralwinding_medium_frac'] > 0.8]
# df3 = df.loc[df['spiralwinding_tight_frac'] > 0.9]
#
# df4 = df.loc[df['bulge_no_bulge_frac'] > 0.9]
# df5 = df.loc[df['bulge_obvious_frac'] > 0.9]
# df6 = df.loc[df['bulge_dominant_frac'] > 0.75]
#
# print(df1.shape)  # 558  469  358
# print(df2.shape)  # 295  163  82
# print(df3.shape)  # 610  419  262
# print(df4.shape)  # 617  529  428
# print(df5.shape)  # 507  348  233
# print(df6.shape)  # 63   35   16
# print(df.shape)   # 3269

