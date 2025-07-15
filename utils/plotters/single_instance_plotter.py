#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from trailer_pose_network.data_setup import TrailerData

SINGLE_INST_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\single_instance\\FF1\\single_instance_ff1.csv"
TRUTH_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\data\\testing\\processed\\FF\\FF1\\FF1_testing.csv"

#%%
# convert to dataframe
single_inst_df = pd.read_csv(SINGLE_INST_CSV)
truth_df = pd.read_csv(TRUTH_CSV, header=None)

#%%
# plot single instances
truth_array = truth_df[truth_df.columns[9]].to_numpy()
truth_array = np.rad2deg(truth_array[:-1])
t = (1/40)*np.arange(0,len(truth_array))
# ax[0].plot(t,single_inst_df["resnet18"])
# ax[0].plot(t,single_inst_df["resnet34"])
# ax[0].plot(t,single_inst_df["mobilenetv2"])
# ax[0].plot(t,single_inst_df["inceptionv3"])
# ax[0].plot(t,single_inst_df["googlenet"])
# ax[0].plot(t,single_inst_df["densenet121"])

ax1 = plt.subplot()
ax1.plot(t,single_inst_df["resnet18"], label="ResNet-18")
ax1.plot(t,single_inst_df["resnet34"], label="ResNet-34")
ax1.plot(t,single_inst_df["densenet121"], label="DenseNet-121")
ax1.plot(t,single_inst_df["mobilenetv2"], label="MobileNetV2")
ax1.plot(t,single_inst_df["googlenet"], c='C8', label="GoogLeNet")
ax1.plot(t,single_inst_df["inceptionv3"], c='C9', label="InceptionV3")
ax1.plot(t,truth_array,  '--', c='k', linewidth=1.5, label="Truth")
ax1.legend(loc="upper left")
ax1.set_ylabel("Articulation Angle (Deg)", fontsize=13,)
ax1.set_xlabel("Time (s)", fontsize=13)
ax1.grid()
ax1.set_axisbelow(True)
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)
plt.show()

# ax1 = plt.subplot(311)
# ax1.plot(t,single_inst_df["resnet18"], label="ResNet-18")
# ax1.plot(t,single_inst_df["resnet34"], label="ResNet-34")
# ax1.plot(t,single_inst_df["densenet121"], label="DenseNet-121")
# ax1.plot(t,single_inst_df["mobilenetv2"], label="MobileNetV2")
# ax1.plot(t,single_inst_df["googlenet"], c='C8', label="GoogLeNet")
# ax1.plot(t,single_inst_df["inceptionv3"], c='C9', label="InceptionV3")
# ax1.plot(t,truth_array,  '--', c='k', linewidth=1.5, label="Truth")
# ax1.legend(loc="outside right upper")
# ax1.set_ylabel("Articulation Angle (Deg)", fontsize=13,)
# ax1.grid()
# ax1.set_axisbelow(True)
# ax1.tick_params(axis='x',labelsize=13)
# ax1.tick_params(axis='y',labelsize=13)

ax2 = plt.subplot(211)
ax2.plot(t,truth_array - single_inst_df["resnet18"], label="ResNet-18")
ax2.plot(t,truth_array - single_inst_df["resnet34"], label="ResNet-34")
ax2.plot(t,truth_array - single_inst_df["densenet121"], label="DenseNet-121")
ax2.plot(t,truth_array - single_inst_df["mobilenetv2"], label="MobileNetV2")
ax2.legend(loc="lower left")
ax2.set_ylabel("Error (Deg)", fontsize=13)
ax2.grid()
ax2.set_axisbelow(True)
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)

ax3 = plt.subplot(212)
ax3.plot(t,truth_array - single_inst_df["googlenet"], c='C8', label="GoogLeNet")
ax3.plot(t,truth_array - single_inst_df["inceptionv3"], c='C9', label="InceptionV3")
ax3.legend(loc="lower left")
ax3.set_ylabel("Error (Deg)", fontsize=13)
ax3.set_xlabel("Time (s)", fontsize=13)
ax3.grid()
ax3.set_axisbelow(True)
ax3.tick_params(axis='x',labelsize=13)
ax3.tick_params(axis='y',labelsize=13)

plt.show()
# plt.tight_layout(pad=3)
# %%
# plot top three: mobilenetv3, densenet121, resnet34
ax1 = plt.subplot(211)
ax1.plot(t,single_inst_df["mobilenetv2"], c='C3', linewidth=2, label="MobileNetV2")
ax1.plot(t,single_inst_df["densenet121"], c='C8', linewidth=2, label="DenseNet-121")
ax1.plot(t,single_inst_df["resnet34"], c='C9', linewidth=2, label="ResNet-34")
ax1.plot(t,truth_array,  '--', c='k', linewidth=1, label="Truth")
ax1.legend(loc="lower left")
ax1.set_ylabel("Articulation Angle (Deg)", fontsize=13,)
ax1.grid()
ax1.set_axisbelow(True)
ax1.tick_params(axis='x',labelsize=13)
ax1.tick_params(axis='y',labelsize=13)

ax2 = plt.subplot(212)
ax2.plot(t,truth_array - single_inst_df["mobilenetv2"], c='C3',label="MobileNetV2")
ax2.plot(t,truth_array - single_inst_df["densenet121"], c='C8',label="DenseNet-121")
ax2.plot(t,truth_array - single_inst_df["resnet34"], c='C9',label="ResNet-34")
ax2.set_ylabel("Error (Deg)", fontsize=13)
ax2.set_xlabel("Time (s)", fontsize=13)
ax2.grid()
ax2.set_axisbelow(True)
ax2.tick_params(axis='x',labelsize=13)
ax2.tick_params(axis='y',labelsize=13)
plt.show()
# %%
