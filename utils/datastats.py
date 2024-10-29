'''
################## Training Data Statistics ##################

Script to analyze training data statistics. Utility tool to
monitor training data distribution before training

##############################################################
'''
#%%
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

#%%
# load data
FULL_CSV = "/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/combined/combined_data.csv"
REDUCED_CSV = "/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/reduced/reduced_data_v2.csv"

full_df= pd.read_csv(FULL_CSV)
reduced_df= pd.read_csv(REDUCED_CSV)

full_cols = full_df.columns
reduced_cols = reduced_df.columns

full_hitch = np.rad2deg(full_df[full_cols[9]].to_numpy())
reduced_hitch = np.rad2deg(reduced_df[reduced_cols[9]].to_numpy())

#%%
# plot
plt.subplot(2,1,1)
plt.plot(full_hitch)
plt.subplot(2,1,2)
plt.plot(reduced_hitch)
plt.tight_layout()
plt.show()

plt.subplot(2,1,1)
plt.hist(full_hitch, bins=15, histtype='bar', ec='black')
plt.ylabel('Frequency')
plt.title('Raw')
plt.subplot(2,1,2)
plt.hist(reduced_hitch, bins=15, histtype='bar', ec='black')
plt.ylabel('Frequency')
plt.xlabel('Hitch Angles [deg]')
plt.title('Proccessed')
plt.tight_layout()
plt.show()


# %%
