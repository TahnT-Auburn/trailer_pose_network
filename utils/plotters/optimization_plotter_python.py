#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

NET = 'densenet121'
CSV_WIDE = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\optuna_wide\\fine_tune_tl\\"+NET+"_optuna_studies_wide_v2.csv"
CSV_NARROW = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\optuna_narrow\\"+NET+"_optuna_studies_narrow_v2.csv"
study_df = pd.read_csv(CSV_WIDE)
narrow_df = pd.read_csv(CSV_NARROW)
# print(study_df["params_lr"].to_list())

# display row with lowest rmse value
min_idx = study_df.loc[:,"value"].idxmin()
print("Optimal params: ")
print(study_df.loc[min_idx,:])

# scatter
x_scatter = np.log10(study_df["params_lr"].to_list())
y_scatter = np.log10(study_df["params_LOSS_SCALE"].to_list())
colors = study_df["value"].to_list()
marker_size = 200

fig, ax = plt.subplots(1,2)
# plt.rcParams['axes.grid'] = True
scatter1 = ax[0].scatter(x_scatter, y_scatter,
                        [50*x for x in study_df["value"].to_list()],
                        c=colors, cmap=plt.cm.coolwarm)
cbar1 = fig.colorbar(scatter1)
cbar1.set_label(label='RMSE', size=15)
cbar1.ax.tick_params(labelsize=13)
ax[0].set_yticks([0,1,2,3])
ax[0].set_ylim([-0.75,3.75])
ax[0].set_xlim([-7.5,-1.5])
ax[0].set_xlabel("Log Learning Rate", fontsize=15)
ax[0].set_ylabel("Log Loss Scale", fontsize=15)
ax[0].set_axisbelow(True)
ax[0].tick_params(axis='x', labelsize=13)
ax[0].tick_params(axis='y', labelsize=13)

# ax[0].grid()
# optimal learning rate scatter
# lr_vals = []
# ls_vals = []
# value_vals = []
# opt_lr = study_df.loc[min_idx,"params_lr"]
# for idx, lr in study_df["params_lr"].items():
#     if lr == opt_lr:
#         lr_vals.append(lr)
#         ls_vals.append(study_df.loc[idx,"params_LOSS_SCALE"])
#         value_vals.append(study_df.loc[idx,"value"])

# scatter2 = ax[1].scatter(np.log10(ls_vals), np.log10(lr_vals), s=[50*x for x in value_vals], c=value_vals, cmap=plt.cm.coolwarm)


# scatter2 = ax[1].scatter(np.log10(ls_vals), np.log10(lr_vals), s=[50*x for x in value_vals], c=value_vals, cmap=plt.cm.coolwarm)
# plt.ylim((np.log10(opt_lr)+0.05,np.log10(opt_lr)-0.05))
x_scatter = narrow_df["params_lr"].to_list()
y_scatter = study_df.loc[min_idx,"params_LOSS_SCALE"]*np.ones(len(narrow_df["params_lr"]))
scatter2 = ax[1].scatter(x_scatter, y_scatter,
                        [50*x for x in narrow_df["value"].to_list()],
                        c=narrow_df["value"].to_list(), cmap=plt.cm.coolwarm)
cbar2 = fig.colorbar(scatter2)
cbar2.set_label(label='RMSE', size=15)
cbar2.ax.tick_params(labelsize=13)
ax[1].set_yticks([study_df.loc[min_idx,"params_LOSS_SCALE"]])
ax[1].set_ylim([study_df.loc[min_idx,"params_LOSS_SCALE"]-5, study_df.loc[min_idx,"params_LOSS_SCALE"]+5])
# ax[1].set_xlim([])
ax[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax[1].set_xlabel("Learning Rate", fontsize=15)
ax[1].set_ylabel("Loss Scale", fontsize=15)
ax[1].set_axisbelow(True)
ax[1].tick_params(axis='x', labelsize=13)
ax[1].tick_params(axis='y', labelsize=13)
# ax[1].grid()
plt.tight_layout(pad=3)
# plt.plot()
plt.show()

#%%
x_scatter = np.log10(study_df["params_lr"].to_list())
y_scatter = np.log10(study_df["params_LOSS_SCALE"].to_list())
colors = study_df["value"].to_list()
fig, ax = plt.subplots()
# plt.rcParams['axes.grid'] = True
scatter1 = ax.scatter(x_scatter, y_scatter,
                        [50*x for x in study_df["value"].to_list()],
                        c=colors, cmap=plt.cm.coolwarm)
cbar1 = fig.colorbar(scatter1)
cbar1.set_label(label='RMSE (Deg)', size=15)
cbar1.ax.tick_params(labelsize=13)
ax.set_yticks([0,1,2,3])
ax.set_ylim([-0.75,3.75])
ax.set_xlim([-7.5,-1.5])
ax.set_xlabel("Log Learning Rate", fontsize=15)
ax.set_ylabel("Log Loss Scale", fontsize=15)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
plt.tight_layout(pad=3)
plt.show()

# %%
fig, ax = plt.subplots()
x_scatter = narrow_df["params_lr"].to_list()
y_scatter = study_df.loc[min_idx,"params_LOSS_SCALE"]*np.ones(len(narrow_df["params_lr"]))
scatter2 = ax.scatter(x_scatter, y_scatter,
                        [50*x for x in narrow_df["value"].to_list()],
                        c=narrow_df["value"].to_list(), cmap=plt.cm.coolwarm)
axins = inset_axes(ax,
                width="100%",  
                height="20%",
                loc='lower center',
                borderpad=-5
                )
cbar2 = fig.colorbar(scatter2, cax=axins, orientation="horizontal")
cbar2.set_label(label='RMSE (Deg)', size=15)
cbar2.ax.tick_params(labelsize=13)
ax.set_yticks([study_df.loc[min_idx,"params_LOSS_SCALE"]])
ax.set_ylim([study_df.loc[min_idx,"params_LOSS_SCALE"]-5, study_df.loc[min_idx,"params_LOSS_SCALE"]+5])
# ax[1].set_xlim([])
ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
ax.set_xlabel("Learning Rate", fontsize=15)
ax.set_ylabel("Loss Scale", fontsize=15)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)
# ax[1].grid()
plt.tight_layout(pad=3)
# plt.plot()
plt.show()