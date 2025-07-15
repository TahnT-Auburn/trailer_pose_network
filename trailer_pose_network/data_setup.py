'''
################### Vanilla Data Setup Script ###################

Script to setup the network intended for use with the TrailerNet.

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

#################################################################
'''
#%%
import os
import torch
import pandas as pd
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from trailer_pose_network.custom_transforms import *

#%% Generate dataset class

class TractorTrailerData(Dataset):
        def __init__(self,
                     csv_file:str,
                     inputs:dict={"cam":True, "can":False, "imu":False},
                     output_states=None,
                     reduce:dict={"target_column":None, "target_size":None},
                     sequential:int=None,
                     sequence_root:str=None,
                     single_test:bool=False,
                     transform=None,
                     cap:list=None,
                     undistort:dict={"cameraMatrices":None, "distCoeffs":None}):
                '''
                Arguments:
                        csv_file (str path): 
                                Path to csv file containing entire training dataset.
                        inputs (dict):
                                Input data to the network. Dictionary with key options "cam", "can", "imu".
                                Default is "cam": True, "can": False, "imu": False.
                        output_states (str or str list): 
                                Names or name list of header columns specifying the individual output states. Default is None.
                        reduce (dict, optional):
                                A dictionary with keys "target_column" and "target_size".\\
                                If passed, data is reduced by a stratification method (uniform fit on target column) to a set target size. 
                        sequential (int, optional):
                                Number of sequential frames to stack. This includes the current frame. Default is None.
                        sequence_root (str path, optional):
                                Root directory to individual sequences. Required field for sequential data. Default is None.
                        single_test (bool, optional):
                                Flag for single instance testing. Slightly modifies how a sequence is pulled. sequence_root must point directly into the single sequence.
                        cap (list, optional): 
                                A two-element list which contains the start and end indices. Default is None.
                        transform (callable, optional): 
                                Transform to be applied to data. Default is None.
                        undistort (dict, optional):
                                Undistorts an image with given camera matrix and distortion coefficients.
                '''
                self.df = pd.read_csv(csv_file, header='infer')
                self.original_df = self.df #create copy of original dataframe for reference
                if cap is not None:
                        self.df = self.df.iloc[cap[0]:cap[1]] # TODO: What is the best way to take in timestamps instead? Also input a dt?
                self.df = self.df.reset_index(drop=True)
                self.column_names = self.df.columns
                self.inputs = inputs
                self.output_states = output_states
                self.reduce = reduce
                self.sequential = sequential
                self.sequence_root = sequence_root
                self.single_test = single_test
                self.transform = transform
                self.undistort = undistort

                # assert sequence root if sequential mode is prompted
                if sequential is not None and sequence_root is None:
                        raise Exception("Sequential mode prompted but sequence root is None.")
                
                # assert valid input keys
                for input in inputs.keys():
                        if input == "cam" or input == "can" or input == "imu":
                                pass
                        else:
                                raise Exception("Invalid keys for inputs.")

                # update a new dataframe that only contains valid entries for a given sequential lookback 
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

                # generate a new dataframe where n past frames are 
                # TODO: Does it save time to preload a new dataframe with all the past
        def __len__(self):
                return len(self.df)
        
        def __getitem__(self, idx):
                #TODO: Generate a temporary dataframe/file with sequential lookbacks in the __init__ fucntion! this should
                #      save A LOT of time!

                # start_time = time.time()
                if self.sequential is not None:
                        # start_time2 = time.time()
                        # Find and match the current data to it's respective index in its sequence
                        current_data = self.df.loc[idx]
                        seq_id = (self.df["SET"][idx], self.df["SUBSET"][idx])
                        current_sequence = self.getSequences(single=True, seq_id=seq_id)
                        seq_idx = current_sequence[current_sequence["LRMC"].isin(current_data)].index
                        
                        # pull a sequential block of data starting from the current data backwards up to N=self.sequential
                        seq_idxes = [i for i in range(seq_idx.item(), seq_idx.item()-self.sequential, -1)]
                        seq_block = current_sequence.iloc[seq_idxes].reset_index(drop=True)
                        # print(f"setup time: {time.time()-start_time2}")

                        # Iterate through seq_block and populate a list of inputs into network.
                        # NOTE: This loop collects a list of all possible inputs (Camera, CAN, and IMU).
                                             
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
                                        STOP=1
                                        concat_images = list(executor.map(cv2.hconcat,image_pairs))
                                        # concat_images = list(executor.map(self.change_img_color,concat_images))
                                        if self.transform is not None:
                                                concat_images = list(executor.map(self.transform,concat_images))

                                input_cam = torch.stack(concat_images)
                                # for image in concat_images:
                                #         image = image.permute(1,2,0).numpy()
                                #         cv2.imshow("test", image)
                                #         cv2.waitKey(0)
                                #         stop=1
                                # input_cam = torch.stack(cam_inputs)

                        if self.inputs["can"]:
                                input_can = torch.stack([torch.tensor(seq_block["steer_ang"].to_list()),torch.tensor(seq_block["vx"].to_list())]).permute(1,0).unsqueeze(dim=1).unsqueeze(dim=2)

                        if self.inputs["imu"]:
                                input_imu = torch.stack([torch.tensor(seq_block["imu_accel_y"].to_list()),torch.tensor(seq_block["imu_gyro_x"].to_list())]).permute(1,0).unsqueeze(dim=1).unsqueeze(dim=2)

          
                        # print(f"threaded image processing time: {time.time()-start_time}")
                        # start_time = time.time()
                        # for i in range(len(seq_block)):
                                
                                # if self.inputs["cam"]:
                                #         
                                #         # left_image = cv2.imread(str(seq_block["LRMC"][i]))
                                #         # right_image = cv2.imread(str(seq_block["RRMC"][i]))
                                #         left_image = left_images[i]
                                #         right_image = right_images[i]
                                #         print(f"image pull time: {time.time()-start_time}")
                                #         # left_image = left_images[i]
                                #         # right_image = right_images[i]
                                #         # undistort left and right images individually
                                #         start_time = time.time()
                                #         if self.undistort["cameraMatrices"] and self.undistort["distCoeffs"] is not None:
                                #                 left_image = self.undistort_image(left_image,
                                #                                                 self.undistort["cameraMatrices"][0],
                                #                                                 self.undistort["distCoeffs"][0])
                                #                 right_image = self.undistort_image(right_image,
                                #                                                 self.undistort["cameraMatrices"][1],
                                #                                                 self.undistort["distCoeffs"][1])
                                        
                                #         # concatenate the images                
                                #         concat_image = cv2.hconcat([right_image, left_image])
                                #         # concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR) #convert color if needed
                                        
                                #         # apply transforms if passed
                                #         if self.transform:
                                #                 concat_image = self.transform(concat_image)
                                #         print(f"Image processing time: {time.time()-start_time}")
                                #         # append all images to a list
                                #         cam_inputs.append(concat_image)
                                        
                                # if self.inputs["can"]:
                                #         # CAN inputs (order: [steer_ang, vx])
                                #         can_inputs.append(torch.tensor([seq_block["steer_ang"][i], seq_block["vx"][i]]))

                                # if self.inputs["imu"]:
                                #         # IMU inputs (order: [accel, gyro])
                                #         # TODO: limit accel and gyro inputs. May not need all 6 measurements for tangent state estimation and nav.
                                #         imu_inputs.append(torch.tensor([seq_block["imu_accel_y"][i], seq_block["imu_gyro_z"][i]]))

                                # imu_inputs.append(torch.tensor([seq_block["imu_accel_x"][i], seq_block["imu_accel_y"][i], seq_block["imu_accel_y"][i],
                                #                                 seq_block["imu_gyro_x"][i], seq_block["imu_gyro_y"][i], seq_block["imu_gyro_z"][i]]))

                                # TODO: Figure out how we want to calculate outputs in this loop. E.g., delta translations, rotations, etc.
                        # print(f"loop time: {time.time()-start_time}")

                        # start_time = time.time()
                        # stack all inputs
                        # if self.inputs["cam"]:
                        #         input_cam = torch.stack(cam_inputs)
                        # if self.inputs["can"]:
                        #         input_can = torch.stack(can_inputs).unsqueeze(dim=1).unsqueeze(dim=2)
                        # if self.inputs["imu"]:
                        #         input_imu = torch.stack(imu_inputs).unsqueeze(dim=1).unsqueeze(dim=2)

                        # generate list of inputs
                        inputs = []
                        if self.inputs["cam"]:
                                inputs.append(input_cam)
                        if self.inputs["can"] and self.inputs["imu"]:
                                inputs.append(torch.cat((input_can,input_imu),dim=3))
                        if self.inputs["can"] and not self.inputs["imu"]:
                                inputs.append(input_can)
                        if self.inputs["imu"] and not self.inputs["can"]:
                                inputs.append(input_imu)

                        # generate outputs by calling customOuptuts function
                        targets = self.customOutputs(seq_block=seq_block)
                        # print(f"wrap up time: {time.time()-start_time}\n")
                        # print(f"Dataloader Time: {time.time()-start_time}")
                        return inputs, targets

                else:
                        left_image = cv2.imread(str(self.df["LRMC"][idx]))
                        right_image = cv2.imread(str(self.df["RRMC"][idx]))

                        # undistort left and right images individually
                        if self.undistort["cameraMatrices"] and self.undistort["distCoeffs"] is not None:
                                left_image = self.undistort_image(left_image,
                                                                self.undistort["cameraMatrices"][0],
                                                                self.undistort["distCoeffs"][0])
                                right_image = self.undistort_image(right_image,
                                                                self.undistort["cameraMatrices"][1],
                                                                self.undistort["distCoeffs"][1])

                        # concatenate the images                
                        input_image = cv2.hconcat([right_image, left_image])
                        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                        
                        # apply transforms if passed
                        if self.transform:
                                input_image = self.transform(input_image)
                        
                        # cv2.imshow('Test', input_image.cpu().numpy().transpose(1, 2, 0))
                        # cv2.waitKey(0)

                        # if a list of states (multi output) is given generate output vector
                        out = []
                        if type(self.output_states) == list:
                                out = [self.df[output_state][idx] for output_state in self.output_states]

                                # for x in self.output_states:
                                #         out.append(self.df[self.column_names[x]][idx])
                        else:
                                out = self.df[self.output_states][idx]
                                # out = self.df[self.column_names[self.output_states]][idx]

                        # convert from rad to deg
                        # out = np.rad2deg(out)
                        # conver to tensor        
                        out = torch.as_tensor(out).squeeze()

                        return input_image, out

        '''
        #########################
        Utility Functions
        #########################
        '''
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
        def change_img_color(self, img):
                '''
                Changes a the color channels of image using OpenCV.
                '''
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                return img
        
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
                x_body = np.cos(seq_block["yaw"].to_numpy()) * seq_block["X"].to_numpy() + np.sin(seq_block["yaw"].to_numpy()) * seq_block["Y"].to_numpy()
                y_body = -np.sin(seq_block["yaw"].to_numpy()) * seq_block["X"].to_numpy() + np.cos(seq_block["yaw"].to_numpy()) * seq_block["Y"].to_numpy()
                
                delX = x_body[0] - x_body[-1]
                delY = y_body[0] - y_body[-1]
                # delX = seq_block["X"].iloc[0] - seq_block["X"].iloc[-1]
                # delY = seq_block["Y"].iloc[0] - seq_block["Y"].iloc[-1]
                delyaw = seq_block["yaw"].iloc[0] - seq_block["yaw"].iloc[-1]
                # hitch = seq_block["hitch"].iloc[0]
                # hitch_rate = seq_block["hitch_rate"][0]

                # out = [delX, delY, delyaw, hitch, hitch_rate]
                out = [delX, delY]
                # out = [hitch, hitch_rate]
                # out = [hitch_rate]
                out = torch.as_tensor(out)

                return out

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

        def undistort_image(self, image, camera_matrix, dist_coeffs):
                """
                Undistort an image using camera matrix and distortion coefficients.
                
                Arguments:
                        image: Input image (distorted)
                        camera_matrix: Camera intrinsic matrix
                        dist_coeffs: Distortion coefficients
                        
                Returns:
                        undistorted: Undistorted image
                """
                h, w = image.shape[:2]
                
                # Refine camera matrix
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
                )
                
                # Undistort
                # undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
                    # Calculate undistortion and rectification maps

                # undistort rectify method
                map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1
                )

                # Apply the maps to undistort the image
                undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)

                # Crop to ROI
                x, y, w_roi, h_roi = roi
                undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
                
                # Resize back to original dimensions
                undistorted_resized = cv2.resize(undistorted_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                undistorted = undistorted_resized

                return undistorted
        
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
# %%
