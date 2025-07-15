'''
Utility script to plot results on data splits
'''
#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# set directories to data
RMSE_CSV_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\splits_results\\rmse"

rmse_dfs = []
for root,_,files in os.walk(RMSE_CSV_PARENT):
    for _, file in enumerate(files):
        rmse_dfs.append((os.path.splitext(file)[0], pd.read_csv(root+'\\'+file)))

# loop through dfs and populate a list of the same splits across networks
day_points = []
night_points = []
mixed_time_points = []
rural_points = []
urban_points = []
mixed_setting_points = []

for idx,rmse_df in enumerate(rmse_dfs):

    day_points.append(float(rmse_dfs[idx][1]['day'].iloc[0]))
    night_points.append(float(rmse_dfs[idx][1]['night'].iloc[0]))
    mixed_time_points.append(float(rmse_dfs[idx][1]['mixed_time'].iloc[0]))

    rural_points.append(float(rmse_dfs[idx][1]['rural'].iloc[0]))
    urban_points.append(float(rmse_dfs[idx][1]['urban'].iloc[0]))
    mixed_setting_points.append(float(rmse_dfs[idx][1]['mixed_setting'].iloc[0]))

reorg_indices = [3,0,5,4,1,2] # for reorganizing the lists to follow a desired order of nets on the plot
cap = None # place cap to only plot selected networks
day_points = list(map(day_points.__getitem__, reorg_indices))[:cap]
night_points = list(map(night_points.__getitem__, reorg_indices))[:cap]
mixed_time_points = list(map(mixed_time_points.__getitem__, reorg_indices))[:cap]
rural_points = list(map(rural_points.__getitem__, reorg_indices))[:cap]
urban_points = list(map(urban_points.__getitem__, reorg_indices))[:cap]
mixed_setting_points = list(map(mixed_setting_points.__getitem__, reorg_indices))[:cap]

#%%
# generate bar plots of rmses for each split
net_names = ['MobileNetV2', 'DenseNet-121', 'ResNet-34', 'ResNet-18', 'GoogleNet', 'Inception-v3'][:cap]
barWidth = 0.2
br1 = np.arange(len(day_points))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
br6 = [x + barWidth for x in br5]

ax1 = plt.subplot(211)
ax1.bar(br1, day_points, width=barWidth, edgecolor='k', color='#ff7f0e',label='Day')
ax1.bar(br2, night_points, width=barWidth, edgecolor='k', color='#1f77b4',label='Night')
ax1.bar(br3, mixed_time_points, width=barWidth, edgecolor='k', color='#2ca02c', label='Mixed')
ax1.set_xticks([r + 2 * barWidth for r in range(len(day_points))], net_names, rotation=-25)
ax1.tick_params(axis='x', labelsize=10)
ax1.tick_params(axis='y', labelsize=13)
ax1.set_ylabel('Time Split RMSEs [Deg]', fontsize=11)
ax1.grid(axis='y')
ax1.set_axisbelow(True)
ax1.legend()

ax2 = plt.subplot(212)
ax2.bar(br1, rural_points, width=barWidth, edgecolor='k', color='#ff7f0e', label='Rural')
ax2.bar(br2, urban_points, width=barWidth, edgecolor='k', color='#1f77b4', label='Urban')
ax2.bar(br3, mixed_setting_points, width=barWidth, edgecolor='k', color='#2ca02c', label='Mixed')
ax2.set_xticks([r + 2 * barWidth for r in range(len(day_points))], net_names, rotation=-25)
ax2.tick_params(axis='x', labelsize=10)
ax2.tick_params(axis='y', labelsize=13)
ax2.set_ylabel('Scene Split RMSEs [Deg]', fontsize=11)
ax2.grid(axis='y')
ax2.set_axisbelow(True)
ax2.legend()

plt.tight_layout()
plt.show()
# %%
