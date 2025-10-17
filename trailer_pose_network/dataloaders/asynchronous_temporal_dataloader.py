import os
import torch
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from concurrent.futures import ThreadPoolExecutor
import time

from torch.utils.data import Dataset

class AsyncTemporalDataLoader(Dataset):
    def __init__(self,
                 sequence_root_processed:str,
                 sequence_root_raw:str,
                 sequential_lookback:int = 2,
                 inputs:dict = {"cam":True, "can":True, "imu":True},
                 reduce:dict={"target_column":None, "target_size":None},
                 single_test:bool = False,
                 transform_img = None,
                 transform_data:bool = False,
                ):
        self.sequence_root_processed = sequence_root_processed
        self.sequence_root_raw = sequence_root_raw
        self.sequential_lookback = sequential_lookback
        self.inputs = inputs
        self.reduce = reduce
        self.single_test = single_test
        self.transform_img = transform_img
        self.transform_data = transform_data

        # assert valid input keys
        for input in inputs.keys():
                if input == "cam" or input == "can" or input == "imu":
                        pass
                else:
                        raise Exception("Invalid keys for inputs.")
        
        # Generate a dataframe from the sequences in the sequence root.
        sequences = self.getSequences(self.sequence_root_processed)
        self.sequence_list = sequences # Save off list of get item
        valid_sequences = []
        # Generate invalid indicies that don't have enough lookback data
        invalid_idxes = [x for x in range(sequential_lookback-1)]            
        for sequence in sequences:
            valid_sequence = sequence.drop(invalid_idxes) # Drop invalid indexes
            valid_sequences.append(valid_sequence)
            
        valid_df = pd.concat(valid_sequences, ignore_index=True,)
        self.df = valid_df

        # Reduce dataframe by stratification if prompted
        if self.reduce["target_column"] is not None and self.reduce["target_size"] is not None:
                self.df = self.stratified_sampling_dataframe(df=self.df,
                                                                target_column=self.reduce["target_column"],
                                                                target_size=self.reduce["target_size"])
        
        # TODO: Add data transform method here for IMU/CAN data. Need to access the raw df so will need to load the full df here
        
        # Get raw list to save off for get item
        self.sequence_list_raw = self.getSequences(self.sequence_root_raw)
        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, idx):
        # get_item_start_time = time.time()
        # Get current datapoint
        current_data = self.df.loc[idx]
        seq_id = (current_data["SET"], current_data["SUBSET"]) # Populate a sequence ID for later use
        
        # Get the current sequence that the current data point belongs to
        for seq in self.sequence_list:
            if seq.iloc[0]["SUBSET"] == current_data["SUBSET"]:
                current_sequence = seq
                break # Stop search
        
        # Get the index that the current data point is in the current sequence by matching a unique key
        seq_idx = current_sequence[current_sequence["LRMC"].isin(current_data)].index
        # Generate a sequential block of data starting from the current data index backwards up to N=self.sequential
        seq_idxes = [i for i in range(seq_idx.item(), seq_idx.item()-self.sequential_lookback, -1)]
        seq_idxes.reverse() # Reverse order back to standard temporal representation
        seq_block = current_sequence.iloc[seq_idxes].reset_index(drop=True) 
    
        # Get the raw data for the current sequence
        for seq in self.sequence_list_raw:
            if seq.iloc[0]["SUBSET"] == current_data["SUBSET"]:
                current_sequence_raw = seq
                break # Stop search
            
        # Find the indices in the raw sequence that match the time values from seq_block
        # NOTE: This works for simulated data where time is perfect.
        # TODO: Find adjustment for experimental data
        t_series = current_sequence_raw['t_clean']
        t_to_find = [seq_block['t_clean'].iloc[0], seq_block['t_clean'].iloc[-1]] # Grab the earliest and latest time from seq_block            
        mask = t_series.isin(t_to_find)
        valid_raw_indices = t_series[mask].index.tolist()
        seq_block_raw = current_sequence_raw.iloc[valid_raw_indices[0]:valid_raw_indices[-1]+1]
        
        # Load images from original seq_block
        if self.inputs["cam"]:
            # start_time = time.time()
            left_images = []
            right_images =[]
            left_paths = seq_block["LRMC"].to_list()
            right_paths = seq_block["RRMC"].to_list()
            with ThreadPoolExecutor(max_workers=32) as executor:
                    left_images = list(executor.map(self.load_image,left_paths))
                    right_images = list(executor.map(self.load_image,right_paths))

                    image_pairs = list(zip(right_images, left_images))
                    concat_images = list(executor.map(cv2.hconcat,image_pairs))
                    # concat_images = list(executor.map(self.change_img_color,concat_images))
                    if self.transform_img is not None:
                            concat_images = list(executor.map(self.transform_img,concat_images))

            input_cam = torch.stack(concat_images)
            # for image in concat_images:
                # image = concat_images[0].permute(1,2,0).numpy()
                # cv2.imshow("test", image)
                # cv2.waitKey(0)
                # image = concat_images[1].permute(1,2,0).numpy()
                # cv2.imshow("tes2", image)
                # cv2.waitKey(0)
                # stop=1
        # Load CAN and IMU from seq_block_raw
        if self.inputs["can"]:
            input_can = torch.stack([torch.tensor(seq_block_raw["steer_ang"].to_list()),torch.tensor(seq_block_raw["vx"].to_list())]).permute(1,0)
        if self.inputs["imu"]:
            input_imu = torch.stack([torch.tensor(seq_block_raw["imu_accel_x"].to_list()),torch.tensor(seq_block_raw["imu_accel_y"].to_list()),torch.tensor(seq_block_raw["imu_accel_z"].to_list()),
            torch.tensor(seq_block_raw["imu_gyro_x"].to_list()),torch.tensor(seq_block_raw["imu_gyro_y"].to_list()),torch.tensor(seq_block_raw["imu_gyro_z"].to_list())]).permute(1,0)
        
        # generate list of inputs
        inputs = []
        if self.inputs["cam"]:
                inputs.append(input_cam)
        if self.inputs["can"] and self.inputs["imu"]:
                inputs.append(torch.cat((input_can,input_imu),dim=1))
        if self.inputs["can"] and not self.inputs["imu"]:
                inputs.append(input_can)
        if self.inputs["imu"] and not self.inputs["can"]:
                inputs.append(input_imu)
        
        # generate outputs by calling customOuptuts function
        targets = self.customOutputs(seq_block=seq_block)
        
        # print(f"Image shape: {inputs[0].shape}")
        # print(f"IMU shape: {inputs[1].shape}")
        # print(f"\nTotal Get item time {time.time() - get_item_start_time}\n")
    
        return inputs, targets
    
    ############# Utility Methods #############
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

    def customOutputs(self, seq_block):
            """
            Function to generate custom outputs from the given csv.
            Currently supports sequential method. Pulls a delta translation X,Y,yaw,
            and a hitch angle given a sequential block of data.

            Arguments:
                    seq_block (dataframe):
                            The current sequential block of data.
            Returns:
                    outputs (torch.tensor):
                            Target outputs.
            """
            pose1 = (seq_block["X"].iloc[-2], seq_block["Y"].iloc[-2], seq_block["yaw"].iloc[-2])
            pose2 = (seq_block["X"].iloc[-1], seq_block["Y"].iloc[-1], seq_block["yaw"].iloc[-1])
            dx_body, dy_body, dyaw = self.tangent_to_body_frame_translation(pose1, pose2)
            
            outputs = [dx_body, dy_body, dyaw]
            outputs = torch.as_tensor(outputs)
            
            return outputs
        
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