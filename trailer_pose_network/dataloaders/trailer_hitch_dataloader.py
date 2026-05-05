import os
import torch
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from concurrent.futures import ThreadPoolExecutor
import time
from torch.utils.data import Dataset

class HitchDataloader(Dataset):
    def __init__(
        self,
        csv_root:str,
        transforms,
        reduce:dict={"target_column":None, "target_size":None},
        single_test:bool=False,
    ):
        self.csv_root = csv_root
        self.transforms = transforms
        self.reduce = reduce
        self.single_test = single_test
        
        # generate full training data from all csvs in root
        sequences = self.getSequences(self.csv_root)
        self.df = pd.concat(sequences, ignore_index=True)
        
        # Reduce dataframe by stratification if prompted
        if self.reduce["target_column"] is not None and self.reduce["target_size"] is not None:
                self.df = self.stratified_sampling_dataframe(
                    df=self.df,
                    target_column=self.reduce["target_column"],
                    target_size=self.reduce["target_size"])
        stop=1
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        left_image = cv2.imread(str(self.df['LRMC'].iloc[idx]))
        right_image = cv2.imread(str(self.df['RRMC'].iloc[idx]))
        input_image = cv2.hconcat([right_image, left_image])
        if self.transforms:
            input_image = self.transforms(input_image)
        
        # Get hitch angle target
        hitch_target = self.df['hitch'].iloc[idx]
        hitch_target = torch.as_tensor(hitch_target)
        
        return input_image, hitch_target.float()
    
    ###### UTILITY FUNCTIONS ######
    def getSequences(self, sequence_root, single:bool=False, seq_id:tuple=None):
        """
        Gets all sequences or a single sequence from the sequnce root directory.
        
        Arguments:
                single (bool, optinal):
                        Flag to pull a single sequence instead of all from root.
                seq_id (tuple, optional):
                        A tuple with form ('SET','SUBSET') that indicates the sequence id.
                        Required field if single is True.
        Returns:
                sequences/sequence (list of dataframes OR dataframe):
                        A dataframe list of sequences or a dataframe for a single sequence.
        """
        if single:
                if seq_id is None:
                        raise Exception ("Single sequence flag was given but seq_id is None.")
                set = seq_id[0]
                subset = seq_id[1]
                #TODO: FIX THIS. 
                seq_path = os.path.join(sequence_root,subset+".csv") if self.single_test else os.path.join(sequence_root,set,subset,subset+".csv") 
                sequence = pd.read_csv(seq_path, dtype={"SUBSET": str}, header='infer')
                return sequence
        else:
                sequences = []
                for root, dirs, files in os.walk(sequence_root):
                        for file in files:
                                if file.endswith(".csv"):
                                        seq_path = os.path.join(root,file)
                                        seq_df = pd.read_csv(seq_path, dtype={"SUBSET": str}, header='infer')
                                        sequences.append(seq_df)
                return sequences
            
    def load_image(self, image_path):
        """Loads an image using OpenCV."""
        try:
                img = cv2.imread(image_path)
                if img is None:
                        print(f"Error: Could not read image at {image_path}")
                        return None
                return img
        except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
            
    def stratified_sampling_dataframe(self, df, target_column, target_size, n_bins=20, random_state=2):
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