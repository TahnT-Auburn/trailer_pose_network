'''
Utility script to generate data based on splits (e.g., Time, Setting, Articulation Angles)
'''
#%%
import os
import numpy as np
import pandas as pd
import csv
import re

TEST_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\combined\\combined_testing_data_augmented.csv"
STATS_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\data_splits\\data_stats.csv"
OUT_CVS_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\splits\\raw"

test_df = pd.read_csv(TEST_CSV, header=None)
stats_df = pd.read_csv(STATS_CSV)

# instaniate new dataframes
df_day = pd.DataFrame()
df_night = pd.DataFrame()
df_mixed_time = pd.DataFrame()
df_rural = pd.DataFrame()
df_urban = pd.DataFrame()
df_mixed_setting = pd.DataFrame()

#search for time and setting sets and orangize into new dataframes
for i in range(0, len(test_df)):
    test_row = test_df.loc[i]
    for j in range(0,len(stats_df)):
        if stats_df['set_type'][j] == "test":
            if re.search(r'\b'+stats_df['set'][j]+r'\b', test_df.loc[i][0]):
                if stats_df['time_setting'][j] == "D":
                    df_day = pd.concat([df_day, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)
                elif stats_df['time_setting'][j]  == "N":
                    df_night = pd.concat([df_night, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)
                elif stats_df['time_setting'][j]  == "M":
                    df_mixed_time = pd.concat([df_mixed_time, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)

                if stats_df['scene_setting'][j]  == "R":
                    df_rural = pd.concat([df_day, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)
                elif stats_df['scene_setting'][j]  == "U":
                    df_urban = pd.concat([df_night, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)
                elif stats_df['scene_setting'][j]  == "M":
                    dr_mixed_setting = pd.concat([df_mixed_time, pd.DataFrame(test_df.loc[i])], axis=1, ignore_index=True)

df_day = df_day.T
df_night = df_night.T
df_mixed_time = df_mixed_time.T
df_rural = df_rural.T
df_urban = df_urban.T
df_mixed_setting = dr_mixed_setting.T

df_day.to_csv(OUT_CVS_PARENT + "\\day.csv", index=False, header=False)
df_night.to_csv(OUT_CVS_PARENT + "\\night.csv", index=False, header=False)
df_mixed_time.to_csv(OUT_CVS_PARENT + "\\mixed_time.csv", index=False, header=False)
df_rural.to_csv(OUT_CVS_PARENT + "\\rural.csv", index=False, header=False)
df_urban.to_csv(OUT_CVS_PARENT + "\\urban.csv", index=False, header=False)
df_mixed_setting.to_csv(OUT_CVS_PARENT + "\\mixed_setting.csv", index=False, header=False)

