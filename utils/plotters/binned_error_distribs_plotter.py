'''
Plotter script for error distributions for binned data
'''
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

#%%
# load data
ERROR_CSV_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\angle_bin_results\\full"
ERROR_CSV_PARENT2 = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\angle_bin_results\\reducedv2"
dfs = []
for root,_,files in os.walk(ERROR_CSV_PARENT):
    for _, file in enumerate(files):
        dfs.append((os.path.splitext(file)[0], pd.read_csv(root+'\\'+file)))

# loop through dfs
net_pos_errors = []
net_neg_errors = []
for idx,df in enumerate(dfs):

# # siphon errors (split between positive and negative values)
    pos_errors = []
    neg_errors = []
    for keys,values in df[1].items():
        if "P" in keys:
            pos_errors.append(np.array(values))
            
        elif "N" in keys:
            neg_errors.append(np.array(values))

    #append set of errors to respective network
    net_pos_errors.append((df[0], pos_errors))
    net_neg_errors.append((df[0], neg_errors))

reorg_indices = [3,0,5,4,1,2]
cap = None
net_pos_errors = list(map(net_pos_errors.__getitem__, reorg_indices))[:cap]
net_neg_errors = list(map(net_neg_errors.__getitem__, reorg_indices))[:cap]

##### for over lay purposes CAN DELETE #####
dfs = []
for root,_,files in os.walk(ERROR_CSV_PARENT2):
    for _, file in enumerate(files):
        dfs.append((os.path.splitext(file)[0], pd.read_csv(root+'\\'+file)))

# loop through dfs
net_pos_errors2 = []
net_neg_errors2 = []
for idx,df in enumerate(dfs):

# # siphon errors (split between positive and negative values)
    pos_errors = []
    neg_errors = []
    for keys,values in df[1].items():
        if "P" in keys:
            pos_errors.append(np.array(values))
            
        elif "N" in keys:
            neg_errors.append(np.array(values))

    #append set of errors to respective network
    net_pos_errors2.append((df[0], pos_errors))
    net_neg_errors2.append((df[0], neg_errors))

reorg_indices = [3,0,5,4,1,2]
cap = None
net_pos_errors2 = list(map(net_pos_errors2.__getitem__, reorg_indices))[:cap]
net_neg_errors2 = list(map(net_neg_errors2.__getitem__, reorg_indices))[:cap]

#%%
# create box plot
net_names = ['MobileNetV2', 'DenseNet-121', 'ResNet-34', 'ResNet-18', 'GoogLeNet', 'Inception-v3'][:cap]

pos_ranges = ['0:10', '10:20', '20:30', '>30']
neg_ranges = ['-0:-10', '-10:-20', '-20:-30', '<-30']

for i in range(0,len(net_pos_errors)):
    plt.figure()
    # print(f'Net: {net_pos_errors[i][0]}')
    ax1 = plt.subplot(211)
    ax1.boxplot(net_pos_errors[i][1], tick_labels=pos_ranges, showfliers=False)
    bplot2_pos = ax1.boxplot(net_pos_errors2[i][1], tick_labels=pos_ranges, showfliers=False, patch_artist=True, widths=0.25)
    for patch in bplot2_pos['boxes']:
        patch.set_facecolor('cyan')
    ax1.set_ylabel('(+) Angle Errors (Deg)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_title(net_names[i], fontsize=15)
    ax2 = plt.subplot(212)
    ax2.boxplot(net_neg_errors[i][1], tick_labels=neg_ranges, showfliers=False)
    bplot2_neg = ax2.boxplot(net_neg_errors2[i][1], tick_labels=neg_ranges, showfliers=False, patch_artist=True, widths=0.25)
    for patch in bplot2_neg['boxes']:
        patch.set_facecolor('cyan')
    ax2.set_ylabel('(-) Angle Errors (Deg)', fontsize=12)
    ax2.set_xlabel('Articulation Angle Ranges (Deg)', fontsize=14)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    plt.show()

# for i in range(0,len(net_pos_errors)):
#     plt.figure()
#     # print(f'Net: {net_pos_errors[i][0]}')
#     ax1 = plt.subplot(211)
#     ax1.boxplot(net_pos_errors[i][1], tick_labels=pos_ranges, showfliers=False)
#     ax1.set_ylabel('(+) Angle Errors (Deg)', fontsize=11)
#     ax1.grid(axis='y', linestyle='--', alpha=0.7)
#     ax1.tick_params(axis='x', labelsize=13)
#     ax1.tick_params(axis='y', labelsize=13)
#     ax1.set_title(net_names[i])
#     ax2 = plt.subplot(212)
#     ax2.boxplot(net_neg_errors[i][1], tick_labels=neg_ranges, showfliers=False)
#     ax2.set_ylabel('(-) Angle Errors (Deg)', fontsize=11)
#     ax2.set_xlabel('Articulation Angle Ranges (Deg)', fontsize=12)
#     ax2.grid(axis='y', linestyle='--', alpha=0.7)
#     ax2.tick_params(axis='x', labelsize=13)
#     ax2.tick_params(axis='y', labelsize=13)
#     plt.tight_layout()
#     plt.show()

# %%
