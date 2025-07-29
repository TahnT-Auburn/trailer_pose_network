'''
utilitiy script to combine csv data sets
'''
#%%
import numpy as np
import pandas as pd 
import os

#%%
# Create combine CSV function
def combineCSVs(parent_dir:str, save_dir:str=None):
    '''
    Combines data from csvs and writes to dataframe.

    Arguments:
        parent_dir (str):
            The root directory with csv files.
        save_dir (str):
            Path to save the combined CSV.
    Returns:
        combined_df (dataframe):
            A dataframe with the combined CSV data.
    '''
    PATHS = []
    for root, dirs, files in os.walk(PARENT):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root,file)
                # print(csv_path)
                PATHS.append(csv_path)
    dfs = []
    for idx,path in enumerate(PATHS):
        # if idx == 0:
        #     header = 'infer'
        # else:
        #     header = 'infer'
        df = pd.read_csv(path, header='infer')
        dfs.append(df)

    # concatentate dataframes
    df_combined = pd.concat(dfs)
    STOP=1
    if save_dir is not None:
        df_combined.to_csv(save_dir, index=False)
        print(f"Saved data to {save_dir}")

#%%
if __name__ == "__main__":
    PARENT = "D:\\TrainingData\\simulation\\processed"
    SAVE = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\data\\simulation\\full_training.csv"
    combineCSVs(PARENT,SAVE)
    