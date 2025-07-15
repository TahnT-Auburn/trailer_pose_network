#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%
# set names
net_names = ["googlenet", "inceptionv3", "resnet18", "resnet34", "densenet121", "mobilenetv2"]

#%%
# load csvs
GOOGLENET_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\googlenet_train_specs.csv"
INCEPTIONV3_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\inceptionv3_train_specs.csv"
RESNET18_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\resnet18_train_specs.csv"
RESNET34_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\resnet34_train_specs.csv"
DENSENET121_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\densenet121_train_specs.csv"
MOBILENETV2_TL = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\finetune_transfer_learning\\reducedv2\\mobilenetv2_train_specs.csv"

#%%
# convert to df
googlenet_tl_df = pd.read_csv(GOOGLENET_TL)
inceptionv3_tl_df = pd.read_csv(INCEPTIONV3_TL)
resnet18_tl_df = pd.read_csv(RESNET18_TL)
resnet34_tl_df = pd.read_csv(RESNET34_TL)
densenet121_tl_df = pd.read_csv(DENSENET121_TL)
mobilenetv2_tl_df = pd.read_csv(MOBILENETV2_TL)

#%%
# plot loss
fig, ax = plt.subplots()
ax.plot(np.arange(1,11), googlenet_tl_df["loss_history"], "->", )
ax.plot(np.arange(1,11),inceptionv3_tl_df["loss_history"], "->")
ax.plot(np.arange(1,11),resnet18_tl_df["loss_history"], "->")
ax.plot(np.arange(1,11),resnet34_tl_df["loss_history"], "->")
ax.plot(np.arange(1,11),densenet121_tl_df["loss_history"], "->")
ax.plot(np.arange(1,11),mobilenetv2_tl_df["loss_history"], "->")
ax.set_xlabel('Epochs', fontsize=15)
ax.set_ylabel('Log Loss', fontsize=15)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.set_xticks(np.arange(1,11))
ax.set_yscale("log")
ax.grid()
ax.set_axisbelow(True)
plt.legend(labels=["GoogLeNet", "Inception-v3",
                    "Resnet-18", "Resnet-34",
                    "Densenet-121", "MobileNetV2"],
            fontsize=10, loc="upper right")
plt.tight_layout(pad=3)
plt.show()

#%%
# plot train rmse
fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(1,11), googlenet_tl_df["rmse_train_history"], "->", )
ax[0].plot(np.arange(1,11),inceptionv3_tl_df["rmse_train_history"], "->")
ax[0].plot(np.arange(1,11),resnet18_tl_df["rmse_train_history"], "->")
ax[0].plot(np.arange(1,11),resnet34_tl_df["rmse_train_history"], "->")
ax[0].plot(np.arange(1,11),densenet121_tl_df["rmse_train_history"], "->")
ax[0].plot(np.arange(1,11),mobilenetv2_tl_df["rmse_train_history"], "->")
# ax[0].set_xlabel('Epochs', fontsize=15)
ax[0].set_ylabel('Train RMSE (Deg)', fontsize=15)
ax[0].tick_params(axis='x', labelsize=13)
ax[0].tick_params(axis='y', labelsize=13)
# ax.set_ylim([1,10])
ax[0].set_xticks(np.arange(1,11))
ax[0].set_yscale("log")
ax[0].set_yticks(np.arange(1,6))
ax[0].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax[0].grid()
ax[0].set_axisbelow(True)
ax[0].legend(labels=["GoogLeNet", "Inception-v3",
                    "Resnet-18", "Resnet-34",
                    "Densenet-121", "MobileNetV2"],
            fontsize=10, loc="upper right")
# plt.tight_layout(pad=3)
# plt.show()

# %%
# plot val rmse
# fig, ax = plt.subplots()
ax[1].plot(np.arange(1,11), googlenet_tl_df["rmse_val_history"], "->", )
ax[1].plot(np.arange(1,11),inceptionv3_tl_df["rmse_val_history"], "->")
ax[1].plot(np.arange(1,11),resnet18_tl_df["rmse_val_history"], "->")
ax[1].plot(np.arange(1,11),resnet34_tl_df["rmse_val_history"], "->")
ax[1].plot(np.arange(1,11),densenet121_tl_df["rmse_val_history"], "->")
ax[1].plot(np.arange(1,11),mobilenetv2_tl_df["rmse_val_history"], "->")
ax[1].set_xlabel('Epochs', fontsize=15)
ax[1].set_ylabel('Val RMSE (Deg)', fontsize=15)
ax[1].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)
# ax.set_ylim([1,10])
ax[1].set_xticks(np.arange(1,11))
ax[1].set_yscale("log")
ax[1].set_yticks(np.arange(1,6))
ax[1].grid()
ax[1].set_axisbelow(True)
# plt.legend(labels=["GoogLeNet", "Inception-v3",
#                     "Resnet-18", "Resnet-34",
#                     "Densenet-121", "MobileNetV2"],
#             fontsize=10, loc="upper right")
ax[1].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
plt.tight_layout(pad=3)
plt.show()