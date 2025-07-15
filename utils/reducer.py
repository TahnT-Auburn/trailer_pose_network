'''
############ Data Reducer ##############
Shuffles and cuts data up to a threshold
########################################
'''

#%%
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer

#%%
# TODO: Make this into a function that can take in a directory and loop through files!
input_path = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\simulation\\training\\full_training.csv"
output_path = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\simulation\\training\\3k_uniform.csv"
write_output = False

df_full = pd.read_csv(input_path, header='infer')

# print(df_full.head())
# full_cols = df_full.columns

# n = df_full[full_cols[16]].value_counts().min()

df_full_shuffled = df_full.sample(frac=1)

# df_reduced = df_full_shuffled.groupby(full_cols[16]).head(5)
# print(full_cols[16])

# def stratified_sampling_dataframe(df, target_column, target_size, n_bins=20, random_state=2):
#     """
#     Perform stratified sampling on a DataFrame to achieve uniform distribution
#     in the target column.
    
#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         The input DataFrame
#     target_column : str
#         The column name to uniformly distribute
#     target_size : int
#         The desired size of the output DataFrame
#     n_bins : int
#         Number of bins to use for stratification
#     random_state : int
#         Random seed for reproducibility
        
#     Returns:
#     --------
#     pandas.DataFrame
#         A subset of the original DataFrame with uniform distribution
#     """
#     # Get column data and reshape for the discretizer
#     column_data = df[target_column].values.reshape(-1, 1)
    
#     # Create uniform-width bins
#     discretizer = KBinsDiscretizer(
#         n_bins=n_bins, 
#         encode='ordinal', 
#         strategy='uniform',
#         subsample=100000  # For very large datasets
#     )
    
#     # Assign bin indices to each data point
#     bin_indices = discretizer.fit_transform(column_data).flatten().astype(int)
    
#     # Create a temporary DataFrame with bin information
#     temp_df = df.copy()
#     temp_df['_bin_index'] = bin_indices
    
#     # Calculate how many samples we want from each bin
#     samples_per_bin = max(1, int(np.ceil(target_size / n_bins)))
    
#     # Sample from each bin
#     sampled_dfs = []
#     for bin_idx in range(n_bins):
#         bin_df = temp_df[temp_df['_bin_index'] == bin_idx]
        
#         if len(bin_df) > 0:
#             # If bin has fewer points than we need, take all of them
#             if len(bin_df) <= samples_per_bin:
#                 sampled_dfs.append(bin_df)
#             else:
#                 # Otherwise randomly sample from this bin
#                 sampled_dfs.append(bin_df.sample(
#                     samples_per_bin, 
#                     random_state=random_state + bin_idx
#                 ))
    
#     # Combine all samples and drop the temporary bin column
#     result_df = pd.concat(sampled_dfs)
#     result_df = result_df.drop('_bin_index', axis=1)
    
#     # Trim to exact target size if we have too many samples
#     if len(result_df) > target_size:
#         result_df = result_df.sample(target_size, random_state=random_state)
    
#     return result_df

def stratified_sampling_dataframe(df, target_column, target_size, n_bins=20, random_state=2):
    """
    Perform stratified sampling on a DataFrame to achieve uniform distribution
    in the target column while returning exactly the target size.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    target_column : str
        The column name to uniformly distribute
    target_size : int
        The desired size of the output DataFrame
    n_bins : int
        Number of bins to use for stratification
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        A subset of the original DataFrame with uniform distribution
        and exactly target_size rows
    """
    if target_size >= len(df):
        return df.copy()
    
    # Get column data and reshape for the discretizer
    column_data = df[target_column].values.reshape(-1, 1)
    
    # Create uniform-width bins
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, 
        encode='ordinal', 
        strategy='uniform',
        subsample=100000  # For very large datasets
    )
    
    # Assign bin indices to each data point
    bin_indices = discretizer.fit_transform(column_data).flatten().astype(int)
    
    # Create a temporary DataFrame with bin information
    temp_df = df.copy()
    temp_df['_bin_index'] = bin_indices
    
    # Calculate base samples per bin and remainder
    base_samples_per_bin = target_size // n_bins
    remainder = target_size % n_bins
    
    # Determine how many samples to take from each bin
    samples_per_bin = [base_samples_per_bin] * n_bins
    
    # Distribute the remainder samples to bins with the most data
    # This helps maintain uniformity while reaching exact target size
    bin_counts = temp_df['_bin_index'].value_counts().sort_values(ascending=False)
    available_bins = bin_counts[bin_counts > 0].index[:remainder]
    
    for bin_idx in available_bins:
        samples_per_bin[bin_idx] += 1
    
    # Sample from each bin
    sampled_dfs = []
    np.random.seed(random_state)
    
    for bin_idx in range(n_bins):
        bin_df = temp_df[temp_df['_bin_index'] == bin_idx]
        target_samples = samples_per_bin[bin_idx]
        
        if len(bin_df) > 0 and target_samples > 0:
            # If bin has fewer points than we need, take all of them
            if len(bin_df) <= target_samples:
                sampled_dfs.append(bin_df)
            else:
                # Otherwise randomly sample from this bin
                sampled_dfs.append(bin_df.sample(
                    target_samples, 
                    random_state=random_state + bin_idx
                ))
    
    # Combine all samples
    if sampled_dfs:
        result_df = pd.concat(sampled_dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame()
    
    # Drop the temporary bin column
    if '_bin_index' in result_df.columns:
        result_df = result_df.drop('_bin_index', axis=1)
    
    # Final check: if we still don't have exactly target_size due to 
    # some bins having insufficient data, adjust accordingly
    if len(result_df) < target_size:
        # Need to sample more - get additional samples from bins with remaining data
        needed = target_size - len(result_df)
        remaining_indices = df.index.difference(result_df.index)
        
        if len(remaining_indices) >= needed:
            additional_samples = df.loc[remaining_indices].sample(
                needed, random_state=random_state + 999
            )
            result_df = pd.concat([result_df, additional_samples], ignore_index=True)
    
    elif len(result_df) > target_size:
        # Need to sample fewer - randomly remove excess
        result_df = result_df.sample(target_size, random_state=random_state + 1000)
    
    return result_df.reset_index(drop=True)

#%%
# call fucntiocn
uniform_state = 'hitch'
data_count = 5000
uniform_df = stratified_sampling_dataframe(df_full_shuffled, uniform_state, data_count, n_bins=20)
# uniform_df = uniform_sample_dataframe(df=df_full_shuffled, column_name='hitch', target_size=data_count)
df_reduced = uniform_df

print("Length of full dataset: %d" % len(df_full))
print("Length of reduced dataset: %d" % len(df_reduced))
print("Percent: %f" % (len(df_reduced)/len(df_full)))

# generate a testing set with equal distribution

# view histograms
full_hitch = np.rad2deg(df_full[uniform_state].to_numpy())
reduced_hitch = np.rad2deg(df_reduced[uniform_state].to_numpy())

sns.set(font_scale=1.5)
sns.set_style("whitegrid")
fig, ax = plt.subplots(2,1)
hist1 = sns.histplot(data={'full_hitch': full_hitch}, x=full_hitch, bins=20, kde=True, ax=ax[0])
hist2 = sns.histplot(data={'red_hitch': reduced_hitch}, x=reduced_hitch, bins=20, kde=True, ax=ax[1])
# hist1.set_ylabel('Frequency', fontsize=20)
# hist3.set_xlabel('Articulation Angles', fontsize=13)s
hist1.set_ylabel('')
hist2.set_ylabel('')
hist1.set_title('Full')
hist2.set_title('Reduced')
fig.supylabel('Frequency', fontsize=20)
fig.supxlabel('Articulation Angles', fontsize=20)
plt.tight_layout()
plt.show()

if write_output:
    df_reduced.to_csv(output_path, index=False)

# %%
