'''
############ Data Reducer ##############
Shuffles and cuts data up to a threshold
########################################
'''


import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv


input_path = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data_augmented.csv'
output_path = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v1.csv'
counter = 0

df = pd.read_csv(input_path, index_col=0)

print(df.head())
column_names = df.columns

n = df[column_names[15]].value_counts().min()

df = df.sample(frac=1)

df = df.groupby(column_names[15]).head(85)
print(column_names[15])

print(len(df.index))

df.to_csv(output_path)
