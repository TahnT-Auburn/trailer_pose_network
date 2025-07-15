'''
################### Sequential Data Setup Script ###################

Script to setup sequential data (e.g., sequential images)

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
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from trailer_pose_network.custom_transforms import *

#%% Generate dataset class

class SeqTrailerData(Dataset):
    def __init__(self,
                 csv_file,
                 output_state,
                 sequence_length=1,
                 spatial_size=(224,224),
                 transform=None,):
                '''
                Arguments:
                csv_file (string): Path to csv file containing entire training dataset
                transform (callable, optional): Transform to be applied to data  
                '''
                self.df = pd.read_csv(csv_file, header=None)
                self.output_state = output_state
                self.sequence_length = sequence_length
                self.spatial_size = spatial_size
                self.column_names = self.df.columns
                self.transform = transform

    def __len__(self):
            return len(self.df)
    
    def __getitem__(self, idx):
        image_sequence = []
        req_pads_collected = False
        requires_padding = False
        
        # collect a list of frames which require padding
        if not req_pads_collected:
                req_pads = []
                for j in range(0,self.sequence_length-1):
                        req_pad = 'LRMC' + str(3+j) + '.jpg'
                        req_pads.append(req_pad)
        req_pads_collected = True
        
        # pad list of frames accordingly if idx matches a frame which requires padding
        current_frame = self.df[self.column_names[0]][idx]
        for index, x in enumerate(req_pads):
                if x in current_frame:
                        requires_padding = True
                        padding_index = index

        if requires_padding:
                # generate pad
                # TODO: Try 1st image padding instead of zero padding
                pad = torch.zeros((self.sequence_length-(padding_index+1),3,self.spatial_size[0],self.spatial_size[1]))
                pad_num = pad.shape[0]
                for k in range(0,padding_index+1):
                        left_image = cv2.imread(str(self.df[self.column_names[0]][idx-k]))
                        right_image = cv2.imread(str(self.df[self.column_names[1]][idx-k]))
                        print(self.df[self.column_names[0]][idx-k])

                        # concatentate left and right images
                        concat_image = cv2.hconcat([right_image, left_image])
                        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)

                        # apply transform if passed    
                        if self.transform:
                                concat_image = self.transform(concat_image)

                        # populate image sequence of images
                        image_sequence.append(concat_image)

                # concatenate pad
                for kk in range(0,pad_num):
                        image_sequence.append(pad[kk])

        else:
                for i in range(0,self.sequence_length):
                        # load left and right image
                        left_image = cv2.imread(str(self.df[self.column_names[0]][idx-i]))
                        right_image = cv2.imread(str(self.df[self.column_names[1]][idx-i]))
                        print(self.df[self.column_names[0]][idx-i])

                        # concatentate left and right images
                        concat_image = cv2.hconcat([right_image, left_image])
                        concat_image = cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR)

                        # apply transform if passed    
                        if self.transform:
                                concat_image = self.transform(concat_image)
                        
                        # ppopulate a sequence of images
                        image_sequence.append(concat_image)
        
        # concatenate image sequence in the channel dimension
        input_image = torch.cat(image_sequence, axis=0)

        # generate target/truth/output state
        target = self.df[self.column_names[self.output_state]][idx]
        target = np.rad2deg(target)
        target = torch.as_tensor(target)

        return image_sequence, input_image, target
# %%
