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
import seaborn as sns

#%%
# load data
FULL_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data.csv"
FULL_CSV_TEST = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\combined\\combined_testing_data.csv"
REDUCED_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v1.csv"
REDUCED_CSV2 = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v2.csv"
STATS = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\data_splits\\data_stats.csv"

full_df= pd.read_csv(FULL_CSV)
full_test_df = pd.read_csv(FULL_CSV_TEST)
reduced_df= pd.read_csv(REDUCED_CSV)
reduced_df2= pd.read_csv(REDUCED_CSV2)

percent = 100*len(reduced_df)/len(full_df)
print("Length of full dataset: %d" % len(full_df))
print("Length of full dataset: %d" % len(reduced_df))
print("Percent: %f" % percent)

full_cols = full_df.columns
full_test_cols = full_test_df.columns
reduced_cols = reduced_df.columns
reduced_cols2 = reduced_df2.columns

full_hitch = np.rad2deg(full_df[full_cols[9]].to_numpy())
full_test_hitch = np.rad2deg(full_test_df[full_test_cols[9]].to_numpy())
reduced_hitch = np.rad2deg(reduced_df[reduced_cols[9]].to_numpy())
reduced_hitch2 = np.rad2deg(reduced_df2[reduced_cols[9]].to_numpy())

#%%
# plot
# plt.subplot(2,1,1)
# plt.plot(full_hitch)
# plt.subplot(2,1,2)
# plt.plot(reduced_hitch)
# plt.tight_layout()
# plt.show()

# plt.subplot(2,1,1)
# plt.hist(full_hitch, bins=15, histtype='bar', ec='black')
# plt.ylabel('Frequency')
# plt.title('Raw')
# plt.subplot(2,1,2)
# plt.hist(reduced_hitch, bins=15, histtype='bar', ec='black')
# plt.ylabel('Frequency')
# plt.xlabel('Hitch Angles [deg]')
# plt.title('Proccessed')
# plt.tight_layout()
# plt.show()

# %%
# seaborn plot
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
fig, ax = plt.subplots(3,1)
hist1 = sns.histplot(data={'full_hitch': full_hitch}, x=full_hitch, bins=20, kde=True, ax=ax[0])
hist2 = sns.histplot(data={'red_hitch2': reduced_hitch2}, x=reduced_hitch, bins=20, kde=True, ax=ax[2])
hist3 = sns.histplot(data={'red_hitch': reduced_hitch}, x=reduced_hitch2, bins=20, kde=True, ax=ax[1])
# hist1.set_ylabel('Frequency', fontsize=20)
# hist3.set_xlabel('Articulation Angles', fontsize=13)
hist1.set_ylabel('')
hist2.set_ylabel('')
hist3.set_ylabel('')
hist1.set_title('100%')
hist2.set_title('20%')
hist3.set_title('50%')
fig.supylabel('Frequency', fontsize=20)
fig.supxlabel('Articulation Angles', fontsize=20)
# plt.show()

#%%
# full training data only
sns.set(font_scale=1.5)
sns.set_style("whitegrid")
fig, ax = plt.subplots(2,1)
hist1 = sns.histplot(data={'full_hitch': full_hitch}, x=full_hitch, bins=20, kde=True, ax=ax[0])
hist2 = sns.histplot(data={'full_test_hitch': full_test_hitch}, x=full_test_hitch, bins=20, kde=True, ax=ax[1])
hist1.set_ylabel('')
hist2.set_ylabel('')
fig.supylabel('Frequency', fontsize=20)
fig.supxlabel('Articulation Angles', fontsize=20)

#%%
# pie chart
day_train = []
mix_time_train = []
night_train = []
rural_train = []
mix_setting_train = []
urban_train = []
day_test = []
mix_time_test = []
night_test = []
rural_test = []
mix_setting_test = []
urban_test = []

stats_df = pd.read_csv(STATS)
for i in range(0,len(stats_df)):
    if stats_df["set_type"][i] == "train":
        if stats_df["time_setting"][i] == 'D':
            day_train.append(stats_df["length"][i])
        elif stats_df["time_setting"][i] == 'N':
            night_train.append(stats_df["length"][i])
        elif stats_df["time_setting"][i] == 'M':
            mix_time_train.append(stats_df["length"][i])

        if stats_df["scene_setting"][i] == 'R':
            rural_train.append(stats_df["length"][i])
        elif stats_df["scene_setting"][i] == 'U':
            urban_train.append(stats_df["length"][i])
        elif stats_df["scene_setting"][i] == 'M':
            mix_setting_train.append(stats_df["length"][i])

    if stats_df["set_type"][i] == "test":
        if stats_df["time_setting"][i] == 'D':
            day_test.append(stats_df["length"][i])
        elif stats_df["time_setting"][i] == 'N':
            night_test.append(stats_df["length"][i])
        elif stats_df["time_setting"][i] == 'M':
            mix_time_test.append(stats_df["length"][i])

        if stats_df["scene_setting"][i] == 'R':
            rural_test.append(stats_df["length"][i])
        elif stats_df["scene_setting"][i] == 'U':
            urban_test.append(stats_df["length"][i])
        elif stats_df["scene_setting"][i] == 'M':
            mix_setting_test.append(stats_df["length"][i])



total_train = np.sum(day_train+night_train+mix_time_train)
total_test = np.sum(day_test+night_test+mix_time_test)
labels = "Night", "Day", "Mixed"
sizes_train = [np.sum(night_train)/total_train, np.sum(day_train)/total_train, np.sum(mix_time_train)/total_train]
sizes_test = [np.sum(night_test)/total_test, np.sum(day_test)/total_test, np.sum(mix_time_test)/total_test]
fig, ax = plt.subplots(2,2)
ax[0,0].pie(sizes_train, labels=labels, autopct='%1.1f%%')
ax[0,0].set_title('Training Set')
ax[0,1].pie(sizes_test, labels=labels, autopct='%1.1f%%')
ax[0,1].set_title('Testing Set')
labels = "Urban", "Rural", "Mixed"
sizes_train = [np.sum(urban_train)/total_train, np.sum(rural_train)/total_train, np.sum(mix_setting_train)/total_train]
sizes_test = [np.sum(urban_test)/total_test, np.sum(rural_test)/total_test, np.sum(mix_setting_test)/total_test]
ax[1,0].pie(sizes_train, labels=labels, autopct='%1.1f%%')
ax[1,1].pie(sizes_test, labels=labels, autopct='%1.1f%%')
# ax[0,0].set_ylabel('Time Splits')
# ax[1,0].set_ylabel('Setting Splits')
plt.show()

# %%
