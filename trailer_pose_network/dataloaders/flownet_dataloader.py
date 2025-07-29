'''
################# Tractor Trailer Image Data Setup for FlowNet #################

Script to set up sequential rear facing camera images compatible with FlowNet

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn 
        
################################################################################
'''
#%%
import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import time
from sklearn.preprocessing import KBinsDiscretizer
from concurrent.futures import ThreadPoolExecutor
import torch
from torch.utils.data import Dataset

class FLownetData(Dataset):
    def __init__(self,
        csv_file:str,
        sequence_root:str,
        sequential:int=2,
        single_test:bool=False,
        reduce:dict={"target_column":None, "target_size":None},
        transform=None):

        self.df = pd.read_csv(csv_file, header="infer")
        self.original_df = self.df
        self.sequence_root = sequence_root
        self.sequential = sequential
        self.single_test = single_test
        self.reduce = reduce
        self.transform = transform

        if sequential is not None:
                sequences = self.getSequences()
                # self.valid_indices = []
                # self.metadata = []
                valid_sequences = []
                invalid_idxes = [x for x in range(sequential-1)]
                for sequence in sequences:
                        # delete invalid rows
                        valid_sequence = sequence.drop(invalid_idxes)
                        valid_sequences.append(valid_sequence)

                valid_df = pd.concat(valid_sequences, ignore_index=True)
                self.df = valid_df #replace for core dataframe

                stop=1
        # reduce dataframe by stratification if prompted
        if self.reduce["target_column"] is not None and self.reduce["target_size"] is not None:
                self.df = self.stratified_sampling_dataframe(df=self.df,
                                                                target_column=self.reduce["target_column"],
                                                                target_size=self.reduce["target_size"])

        stop=1


    def __len__(self):
        return len(self.df)


    def __getitem__(self,idx):
        # start_time = time.time()
        # Find and match the current data to it's respective index in its sequence
        current_data = self.df.loc[idx]
        seq_id = (self.df["SET"][idx], self.df["SUBSET"][idx])
        current_sequence = self.getSequences(single=True, seq_id=seq_id)
        seq_idx = current_sequence[current_sequence["LRMC"].isin(current_data)].index
        
        # pull a sequential block of data starting from the current data backwards up to N=self.sequential
        #TODO: Look into adjusting for longer ranges (Note that FlowNet just takes two images)
        seq_idxes = [i for i in range(seq_idx.item(), seq_idx.item()-self.sequential, -1)]
        seq_block = current_sequence.iloc[seq_idxes].reset_index(drop=True)
        seq_block = seq_block.iloc[[0,-1]].reset_index(drop=True) # filters two images (latest and oldest)
        seq_block = seq_block[::-1] # reverse order making the older frame the first image (NOTE: Not sure how much this affects things!)
        
        left_images = []
        right_images =[]
        left_paths = seq_block["LRMC"].to_list()
        right_paths = seq_block["RRMC"].to_list()       
        with ThreadPoolExecutor(max_workers=16) as executor:
                left_images = list(executor.map(self.load_image,left_paths))
                right_images = list(executor.map(self.load_image,right_paths))

                image_pairs = list(zip(right_images, left_images))
                STOP=1
                concat_images = list(executor.map(cv2.hconcat,image_pairs))
                # concat_images = list(executor.map(self.change_img_color,concat_images))
                if self.transform is not None:
                        concat_images = list(executor.map(self.transform,concat_images)) 
        
        # package inputs and retrieve targets
        input_images = torch.cat(concat_images, dim=0)
        input_can = torch.stack([torch.tensor(seq_block["steer_ang"].to_list()),torch.tensor(seq_block["vx"].to_list())]).permute(1,0)
        input_imu = torch.stack([torch.tensor(seq_block["imu_accel_x"].to_list()),torch.tensor(seq_block["imu_accel_y"].to_list()),torch.tensor(seq_block["imu_accel_z"].to_list()),
                                 torch.tensor(seq_block["imu_gyro_x"].to_list()),torch.tensor(seq_block["imu_gyro_y"].to_list()),torch.tensor(seq_block["imu_gyro_z"].to_list())]).permute(1,0)
        input_inert = torch.cat((input_can,input_imu),dim=1)
        
        inputs = [input_images, input_inert]
        targets = self.customOutputs(seq_block=seq_block)

        stop=1
        # print(f"getitem time:", {time.time()-start_time})
        return inputs, targets
                
                
                
                

    '''
    ##################
    Utility functions
    ##################
    '''
    def getSequences(self, single:bool=False, seq_id:tuple=None):
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
                seq_path = os.path.join(self.sequence_root,subset+".csv") if self.single_test else os.path.join(self.sequence_root,set,subset,subset+".csv") 
                sequence = pd.read_csv(seq_path, header='infer')

                return sequence
        else:
                sequences = []
                for root, dirs, files in os.walk(self.sequence_root):
                        for file in files:
                                if file.endswith(".csv"):
                                        seq_path = os.path.join(root,file)
                                        seq_df = pd.read_csv(seq_path, header='infer')
                                        sequences.append(seq_df)
                
                return sequences
            
    
    def tangent_to_body_frame_translation(self, pose1, pose2):
        """
        Convert from tangent plane poses to body frame relative translation
        
        Args:
            pose1: (X1, Y1, yaw1) - starting pose
            pose2: (X2, Y2, yaw2) - ending pose
            
        Returns:
            (dx_body, dy_body) - translation in body frame of pose1
        """
        X1, Y1, yaw1 = pose1
        X2, Y2, yaw2 = pose2
        
        # World frame displacement
        dx_world = X2 - X1
        dy_world = Y2 - Y1
        
        # Rotate into body frame of pose1
        cos_yaw = np.cos(-yaw1)
        sin_yaw = np.sin(-yaw1)
        
        dx_body = cos_yaw * dx_world - sin_yaw * dy_world
        dy_body = sin_yaw * dx_world + cos_yaw * dy_world
        
        # Relative yaw change
        dyaw = yaw2 - yaw1
        
        # Normalize yaw to [-pi, pi]
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        
        return dx_body, dy_body, dyaw
            
    def customOutputs(self, seq_block):
            """
            Function to generate custom outputs from the given csv.
            Currently supports sequential method. Pulls a delta translation X,Y,yaw,
            and a hitch angle given a sequential block of data.

            Arguments:
                    seq_block (dataframe):
                            The current sequential block of data.
            Returns:
                    outs (list):
                            A list of output tensors.
            """
            # seq_block = seq_block[::-1] # reverse order again
            
            # x_body = np.cos(seq_block["yaw"].to_numpy()) * seq_block["X"].to_numpy() + np.sin(seq_block["yaw"].to_numpy()) * seq_block["Y"].to_numpy()
            # y_body = -np.sin(seq_block["yaw"].to_numpy()) * seq_block["X"].to_numpy() + np.cos(seq_block["yaw"].to_numpy()) * seq_block["Y"].to_numpy()
            
            # delX = x_body[0] - x_body[-1]
            # delY = y_body[0] - y_body[-1]
            # # delX = seq_block["X"].iloc[0] - seq_block["X"].iloc[-1]
            # # delY = seq_block["Y"].iloc[0] - seq_block["Y"].iloc[-1]
            # delyaw = seq_block["yaw"].iloc[0] - seq_block["yaw"].iloc[-1]
            hitch = seq_block["hitch"].iloc[-1]
            hitch_rate = seq_block["hitch_rate"][-1]

            pose1 = (seq_block["X"].iloc[0], seq_block["Y"].iloc[0], seq_block["yaw"].iloc[0])
            pose2 = (seq_block["X"].iloc[1], seq_block["Y"].iloc[1], seq_block["yaw"].iloc[1])
            dx_body, dy_body, dyaw = self.tangent_to_body_frame_translation(pose1, pose2)
            
            # out = [delX, delY, delyaw, hitch, hitch_rate]
            out = [dx_body, dy_body]
            # out = [hitch, hitch_rate]
            # out = [hitch_rate]
            out = torch.as_tensor(out)

            return out
        
                    
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