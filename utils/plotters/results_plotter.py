#%%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%
# load data 
CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\testset_rmse\\testset_rmse_vs_size.csv"
df = pd.read_csv(CSV)

#%%
# plot parameters vs rmse
fig, ax = plt.subplots()
x_scatter = df["parameters"].to_numpy()
y_scatter = df["test_rmse_full"].to_numpy()
scatter = ax.scatter(x_scatter, y_scatter,
           [2e3*x/df["parameters"].max() for x in df["parameters"].to_list()],
           c=df["parameters"], edgecolors='white')
ax.set_xlim([0e7,3.0e7])
ax.set_ylim([0,8])
ax.set_xlabel('Parameters', fontsize=13)
ax.set_ylabel('Test RMSE (Deg)', fontsize=13)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()

cbar = fig.colorbar(scatter, location='bottom')
cbar.set_ticks([])
ax.set_axisbelow(True)
fig.tight_layout(pad=3)
# plt.show()

#%%
# plot gflops vs rmse
fig, ax = plt.subplots()
x_scatter = df["gflops"].to_numpy()
y_scatter = df["test_rmse_full"].to_numpy()
scatter = ax.scatter(x_scatter, y_scatter,
           [2e3*x/df["parameters"].max() for x in df["parameters"].to_list()],
           c=df["parameters"], edgecolors='white')
ax.set_xlim([0,6.5])
ax.set_ylim([0,8])
ax.set_xlabel('GFLOPs', fontsize=13)
ax.set_ylabel('Test RMSE (Deg)', fontsize=13)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()

cbar = fig.colorbar(scatter, location='right')
cbar.set_label("Parameters", fontsize=13)
# cbar.set_ticks([])
ax.set_axisbelow(True)
fig.tight_layout(pad=3)
# plt.show()

# %%
# barplot
df_sorted = df.sort_values("test_rmse_full")
df_sorted.reset_index(drop=True)
fig, ax = plt.subplots()
x = np.arange(0,len(df_sorted["net"]))+1
x_labels = df_sorted["name"].to_list()
# x = df_sorted["gflops"].to_numpy()
y = df_sorted["test_rmse_full"].to_numpy()
colors = [x/df_sorted["gflops"].max() for x in df_sorted["gflops"]]
colormap = plt.cm.get_cmap('RdBu',20)
norm = mpl.colors.Normalize(vmin=0, vmax=df_sorted["gflops"].max())
bar = ax.bar(x,y, color=colormap(colors), tick_label=x_labels, edgecolor='k')
ax.set_ylabel("Test RMSE (Deg)", fontsize=13)
# ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.set_xticklabels(x_labels, rotation = -25)
# ax.set_xticks(x,x_labels)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm),  ax=plt.gca())
cbar.set_label("GFLOPs", fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.grid(axis='y')
ax.set_axisbelow(True)
# for i, v in enumerate(y):
#     plt.text(x[i] - 0.25, v + 0.01, str(round(df_sorted["gflops"][i],3)))
# plt.show()

# %%
# barplot
df_sorted = df.sort_values("test_rmse_full")
df_sorted.reset_index(drop=True)
fig, ax = plt.subplots()
x = np.arange(0,len(df_sorted["net"]))+1
x_labels = df_sorted["name"].to_list()
# x = df_sorted["gflops"].to_numpy()
y = df_sorted["test_rmse_full"].to_numpy()
colors = [x/df_sorted["parameters"].max() for x in df_sorted["parameters"]]
colormap = plt.cm.get_cmap('RdBu',20)
norm = mpl.colors.Normalize(vmin=0, vmax=df_sorted["parameters"].max())
bar = ax.bar(x,y, color=colormap(colors), tick_label=x_labels, edgecolor='k')
ax.set_xticklabels(x_labels, rotation = -25)
ax.set_ylabel("Test RMSE (Deg)", fontsize=13)
# ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# ax.set_xticks(x,x_labels)
cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=colormap, norm=norm),  ax=plt.gca())
cbar.set_label("Parameters", fontsize=13)
cbar.ax.tick_params(labelsize=13)
ax.grid(axis='y')
ax.set_axisbelow(True)
# for i, v in enumerate(y):
#     plt.text(x[i] - 0.25, v + 0.01, str(round(df_sorted["gflops"][i],3)))
# plt.show()

#%%
# plot size vs training time
# fig, ax = plt.subplots(2,1)
# x_scatter = df["gflops"].to_numpy()[:6]
# y_scatter = df["training_time_full"].to_numpy()[:6]
# y_scatter = y_scatter / 60
# scatter = ax[0].scatter(x_scatter, y_scatter,
#            [2e3*x/df["gflops"].max() for x in df["gflops"].to_list()[:6]],
#            c=["r","b","g","c","m","y"], edgecolors='white')
# ax[0].set_xlim([0,6.5])
# ax[0].set_ylim([125,230])
# ax[0].set_xlabel('GFLOPs', fontsize=13)
# # ax[0].set_ylabel('Training Time (min)', fontsize=13)
# ax[0].tick_params(axis='x', labelsize=13)
# ax[0].tick_params(axis='y', labelsize=13)
# ax[0].grid()

# # cbar = fig.colorbar(scatter, location='right')
# # cbar.set_label("Parameters", fontsize=13)
# # cbar.set_ticks([])
# ax[0].set_axisbelow(True)
# # fig.tight_layout(pad=3)

# x_scatter = df["parameters"].to_numpy()[:6]
# y_scatter = df["training_time_full"].to_numpy()[:6]
# y_scatter = y_scatter / 60
# scatter = ax[1].scatter(x_scatter, y_scatter,
#            [2e3*x/df["parameters"].max() for x in df["parameters"].to_list()[:6]],
#            c=["r","b","g","c","m","y"], edgecolors='white')
# ax[1].set_xlim([0,3e7])
# ax[1].set_ylim([125,230])
# ax[1].set_xlabel('Parameters', fontsize=13)
# # ax[1].set_ylabel('Training Time (min)', fontsize=13)
# ax[1].tick_params(axis='x', labelsize=13)
# ax[1].tick_params(axis='y', labelsize=13)
# ax[1].grid()

# # cbar = fig.colorbar(scatter, location='right')
# # cbar.set_label("Parameters", fontsize=13)
# # cbar.set_ticks([])
# ax[1].set_axisbelow(True)
# fig.tight_layout(pad=3)
# fig.supylabel('Frequency', fontsize=13)
# plt.show()
# %%
fig, ax = plt.subplots()
x_scatter = df["gflops"].to_numpy()
y_scatter = df["training_time_full"].to_numpy()
y_scatter = y_scatter / 60
scatter = ax.scatter(x_scatter, y_scatter,
           [2e3*x/df["parameters"].max() for x in df["parameters"].to_list()],
           c=df["parameters"], edgecolors='white')
# for idx, name in df["name"].items():
#         ax.text(x_scatter[idx],y_scatter[idx]-0.1, name, fontsize=10, color='k')
ax.set_xlim([0,6.5])
ax.set_ylim([150,220])
ax.set_xlabel('GFLOPs', fontsize=13)
ax.set_ylabel('Training Time (min)', fontsize=13)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()

cbar = fig.colorbar(scatter, location='right')
cbar.set_label("Parameters", fontsize=13)
# cbar.set_ticks([])
ax.set_axisbelow(True)
fig.tight_layout(pad=3)
plt.show()

# %%
fig, ax = plt.subplots()
x_scatter = df["gflops"].to_numpy()
y_scatter = df["inference_time"].to_numpy()
y_scatter = y_scatter * 1e3
scatter = ax.scatter(x_scatter, y_scatter,
           [2e3*x/df["parameters"].max() for x in df["parameters"].to_list()],
           c=df["parameters"], edgecolors='white')
# for idx, name in df["name"].items():
#         ax.text(x_scatter[idx],y_scatter[idx]-0.1, name, fontsize=10, color='k')
ax.set_xlim([0,6.5])
ax.set_ylim([18.5,27.5])
ax.set_xlabel('GFLOPs', fontsize=13)
ax.set_ylabel('Inference Time (ms)', fontsize=13)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
ax.grid()

cbar = fig.colorbar(scatter, location='right')
cbar.set_label("Parameters", fontsize=13)
# cbar.set_ticks([])
ax.set_axisbelow(True)
fig.tight_layout(pad=3)
plt.show()

# %%
