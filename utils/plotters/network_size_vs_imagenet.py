#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd

CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\cnn_imagenet_specs\\cnn_imagenet_specs.csv"
df = pd.read_csv(CSV)
# set global plotting parameters
# mpl.rcParams['axes.labelsize'] = 10

# parameters vs top-1 acc
fig, ax = plt.subplots()
# ax2 = ax.twinx()
x_scatter = df["gflops"].to_numpy()
y_scatter = df["top_1_acc"].to_numpy()
name = df["name"].to_list()
scatter = ax.scatter(x_scatter, (y_scatter),
             [2e3*(x/df["parameters"].max()) for x in df["parameters"].to_list()],
             c=df["parameters"], edgecolors='white')


# plt.text(x_scatter[0]+1e8,100-y_scatter[0]-0.1, name[0], fontsize=5)
net_label_size = 9
for idx, name in df["name"].items():
#     #
#     if idx in np.arange(0,6,1):
#         font_color = 'g'
#     else:
#         font_color = 'k'

#     # adjust label locations
#     if df["net"][idx] == 'mobilenetv2':
#         ax.text(x_scatter[idx]+6e8,100-y_scatter[idx]-0.17, name, fontsize=net_label_size, color=font_color)
#     elif df["net"][idx] == 'alexnet':
#         ax.text(x_scatter[idx]+9e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color=font_color)
#     elif df["net"][idx] == 'densenet169':
#         ax.text(x_scatter[idx]-0.2e10,100-y_scatter[idx]-1.2, name, fontsize=net_label_size, color=font_color)
#     elif df["net"][idx] == 'vgg16':
#         ax.text(x_scatter[idx]-.11e10,100-y_scatter[idx]-0.0, name, fontsize=net_label_size, color=font_color)
#     elif df["net"][idx] == 'vgg19':
#         ax.text(x_scatter[idx]-.11e10,100-y_scatter[idx]-0.0, name, fontsize=net_label_size, color=font_color)
#     else:
#         ax.text(x_scatter[idx]+8e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color=font_color)
#         ax.text(x_scatter[idx]+8e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color=font_color)

    ax.text(x_scatter[idx],y_scatter[idx]-0.1, name, fontsize=net_label_size, color='k')

ax.set_xlim([-1,25])
ax.set_ylim([55,85])
ax.set_xlabel("GFLOPs", fontsize=15)
ax.set_ylabel("Top-1% Accuracy", fontsize=15)
ax.set_axisbelow(True)
ax.set_xticks(np.arange(0,25,5))
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
fig.tight_layout(pad=3)
cbar = fig.colorbar(scatter)
cbar.set_label(label='Parameters', size=15)
cbar.ax.tick_params(labelsize=11)

# plt.ylim([55,80])
plt.grid()
plt.show()

#%%
# plot selected networks only
fig, ax = plt.subplots()
net_label_size = 10
cap = np.arange(0,6)
x_scatter = df["gflops"][cap].to_numpy()
y_scatter = df["top_1_acc"][cap].to_numpy()
name = df["name"][cap].to_list()
scatter = ax.scatter(x_scatter, (y_scatter),
             [1e3*(x/df["parameters"][cap].max()) for x in df["parameters"][cap].to_list()],
             c=df["parameters"][cap], edgecolors='white')

for idx, name in df["name"][cap].items():
        ax.text(x_scatter[idx],y_scatter[idx]-0.1, name, fontsize=net_label_size, color='k')

ax.set_xlim([0,7.6])
ax.set_ylim([68,80])
ax.set_xlabel("GFLOPs", fontsize=15)
ax.set_ylabel("Top-1% Accuracy", fontsize=15)
ax.set_axisbelow(True)
ax.set_xticks(np.arange(0,7.6,2.5))
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
fig.tight_layout(pad=3)
cbar = fig.colorbar(scatter)
cbar.set_label(label='Parameters', size=5)
cbar.ax.tick_params(labelsize=13)
plt.grid()
plt.show()

# %%
# plot subplot
plt.rcParams['axes.grid'] = True
# selected nets
fig, ax = plt.subplots(2,1)
net_label_size = 15
cap = np.arange(0,6)
x_scatter = df["gflops"][cap].to_numpy()
y_scatter = df["top_1_acc"][cap].to_numpy()
name = df["name"][cap].to_list()
scatter = ax[0].scatter(x_scatter, (y_scatter),
             [2e3*(x/df["parameters"][cap].max()) for x in df["parameters"][cap].to_list()],
             c=df["parameters"][cap], edgecolors='white')

# for idx, name in df["name"][cap].items():
#         ax[0].text(x_scatter[idx]+8e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color='k')

ax[0].set_xlim([0,7.6])
ax[0].set_ylim([68,80])
ax[0].set_xlabel("GFLOPs", fontsize=13)
ax[0].set_ylabel("Top-1% Accuracy on ImageNet", fontsize=13)
ax[0].set_axisbelow(True)
ax[0].set_xticks(np.arange(0,7.6,2.5))
ax[0].tick_params(axis='x', labelsize=13)
ax[0].tick_params(axis='y', labelsize=13)
fig.tight_layout(pad=3)
cbar = fig.colorbar(scatter)
cbar.set_label(label='Parameters', size=13)
cbar.ax.tick_params(labelsize=13)

# all nets
# fig, ax = plt.subplots(1,2)
# ax2 = ax.twinx()
x_scatter = df["gflops"].to_numpy()
y_scatter = df["top_1_acc"].to_numpy()
name = df["name"].to_list()
scatter = ax[1].scatter(x_scatter, (y_scatter),
             [2e3*(x/df["parameters"].max()) for x in df["parameters"].to_list()],
             c=df["parameters"], edgecolors='white')


# plt.text(x_scatter[0]+1e8,100-y_scatter[0]-0.1, name[0], fontsize=5)
net_label_size = 12
# for idx, name in df["name"].items():
    #
    # if idx in np.arange(0,6,1):
    #     font_color = 'g'
    # else:
    #     font_color = 'k'

    # # adjust label locations
    # if df["net"][idx] == 'mobilenetv2':
    #     ax[1].text(x_scatter[idx]+6e8,100-y_scatter[idx]-0.17, name, fontsize=net_label_size, color=font_color)
    # elif df["net"][idx] == 'alexnet':
    #     ax[1].text(x_scatter[idx]+9e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color=font_color)
    # elif df["net"][idx] == 'densenet169':
    #     ax[1].text(x_scatter[idx]-0.2e10,100-y_scatter[idx]-1.2, name, fontsize=net_label_size, color=font_color)
    # elif df["net"][idx] == 'vgg16':
    #     ax[1].text(x_scatter[idx]-.11e10,100-y_scatter[idx]-0.0, name, fontsize=net_label_size, color=font_color)
    # elif df["net"][idx] == 'vgg19':
    #     ax[1].text(x_scatter[idx]-.11e10,100-y_scatter[idx]-0.0, name, fontsize=net_label_size, color=font_color)
    # else:
    #     ax[1].text(x_scatter[idx]+8e8,100-y_scatter[idx]-0.1, name, fontsize=net_label_size, color=font_color)

ax[1].set_xlim([-1,25])
ax[1].set_ylim([50,85])
ax[1].set_xlabel("GFLOPs", fontsize=13)
ax[1].set_ylabel("Top-1% Accuracy on ImageNet", fontsize=13)
ax[1].set_axisbelow(True)
ax[1].set_xticks(np.arange(0,25,5))
ax[1].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)
fig.tight_layout(pad=3)
cbar = fig.colorbar(scatter)
cbar.set_label(label='Parameters', size=13)
cbar.ax.tick_params(labelsize=13)

plt.show()

# %%
