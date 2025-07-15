'''
Utility script to generate a dataset with evenly populated
bins with respect to a range of desired articulation angles.
'''
#%%
import numpy as np
import pandas as pd

TEST_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\combined\\combined_testing_data_augmented.csv"
OUT_CSV_PARENT = "C:\\Users\\pzt0029\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\binned\\"

write_csvs = True
# def custom_df_binning(df, value_column, desired_counts, desired_ranges=None):
#     """
#     Bin DataFrame data with specified counts per bin while respecting desired ranges when possible
    
#     Parameters:
#     df - pandas DataFrame containing the data
#     value_column - string, name of the column to bin
#     desired_counts - list of integers, desired number of points in each bin
#     desired_ranges - list of tuples (min, max) for each bin, optional
    
#     Returns:
#     binned_dfs - list of DataFrames containing the binned data
#     bin_edges - list of tuples with actual (min, max) for each bin
#     """
#     # Sort the DataFrame by the value column
#     sorted_df = df.sort_values(by=value_column).reset_index(drop=True)
#     total_points = len(sorted_df)
#     total_desired = sum(desired_counts)
    
#     if total_desired > total_points:
#         raise ValueError("Sum of desired counts exceeds available data points")
    
#     binned_dfs = []
#     bin_edges = []
#     start_idx = 0
    
#     for i, count in enumerate(desired_counts):
#         if start_idx >= total_points:
#             break
            
#         end_idx = min(start_idx + count, total_points)
#         bin_df = sorted_df.iloc[start_idx:end_idx].copy()
        
#         # Check if we need to respect specific range constraints
#         if desired_ranges and i < len(desired_ranges):
#             min_val, max_val = desired_ranges[i]
#             # Filter points that fall within the desired range
#             bin_df = bin_df[(bin_df[value_column] >= min_val) & 
#                             (bin_df[value_column] <= max_val)]
            
#         if not bin_df.empty:
#             binned_dfs.append(bin_df)
#             bin_edges.append((bin_df[value_column].min(), bin_df[value_column].max()))
        
#         start_idx = end_idx
    
#     return binned_dfs, bin_edges

def custom_df_binning(df, value_column, desired_counts, desired_ranges):
    """
    Bin DataFrame data with specified counts per bin while respecting desired ranges
    
    Parameters:
    df - pandas DataFrame containing the data
    value_column - string, name of the column to bin
    desired_counts - list of integers, desired number of points in each bin
    desired_ranges - list of tuples (min, max) for each bin
    
    Returns:
    binned_dfs - list of DataFrames containing the binned data
    bin_edges - list of tuples with actual (min, max) for each bin
    """
    # Create a copy of the input dataframe to avoid modifying it
    df_copy = df.copy()
    
    binned_dfs = []
    bin_edges = []
    
    # Process each desired bin
    for i, (count, (min_val, max_val)) in enumerate(zip(desired_counts, desired_ranges)):
        # Filter rows that fall within the current range
        mask = (df_copy[value_column] >= min_val) & (df_copy[value_column] <= max_val)
        range_df = df_copy[mask].copy()
        
        if len(range_df) == 0:
            # No data in this range
            binned_dfs.append(pd.DataFrame(columns=df.columns))
            bin_edges.append((min_val, max_val))
            continue
            
        if len(range_df) <= count:
            # Not enough data in range, take all available
            bin_df = range_df
        else:
            # Too much data in range, sample the desired count
            bin_df = range_df.sample(count, random_state=42)
            
        # Remove selected rows from original dataframe to prevent reuse
        df_copy = df_copy.drop(bin_df.index)
        
        binned_dfs.append(bin_df)
        
        # Calculate actual bin edges from the data
        if not bin_df.empty:
            actual_min = bin_df[value_column].min()
            actual_max = bin_df[value_column].max()
            bin_edges.append((actual_min, actual_max))
        else:
            bin_edges.append((min_val, max_val))
    
    return binned_dfs, bin_edges

df_full = pd.read_csv(TEST_CSV, names = [str(x) for x in range(0,17)])
df_shuffled = df_full.sample(frac=1).reset_index(drop=True)

num_bins = 8
desired_count = 300
binned_dfs, bin_edges = custom_df_binning(df_shuffled, '16',
                                           desired_counts=[desired_count]*num_bins,
                                           desired_ranges=[(-50,-31), (-30, -21), (-20, -11), (-10,-1),
                                                           (31,50), (21,30), (11,20), (0,10)])

names = ['50_30_N', '30_20_N', '20_10_N', '10_0_N', '50_30_P', '30_20_P', '20_10_P', '10_0_P']
if write_csvs:
    for idx, binned_df in enumerate(binned_dfs):
        binned_df.reset_index(drop=True).to_csv(OUT_CSV_PARENT + names[idx] + '.csv', index=False, header=False)