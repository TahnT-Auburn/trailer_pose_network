'''
################### Trailer Network Network Setup Script ###################

Script to setup the network intended for use with the TrailerNet.

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

############################################################################
'''
#%%
import os
import torch
import pandas as pd
from skimage import io, transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from custom_transforms import *

#%% Generate dataset class

class TrailerData(Dataset):
        def __init__(self, csv_file, transform=None):
                '''
                Arguments:
                csv_file (string): Path to csv file containing entire training dataset
                transform (callable, optional): Transform to be applied to data  
                '''
                self.df = pd.read_csv(csv_file)
                self.transform = transform
                self.column_names = self.df.columns

        def __len__(self):
                return len(self.df)
        
        def __getitem__(self, idx):
                # read in camera images at given idx
                left_image = cv2.imread(str(self.df[self.column_names[0]][idx]))
                right_image = cv2.imread(str(self.df[self.column_names[1]][idx]))

                # concatenate the images
                input_image = cv2.hconcat([right_image, left_image])
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
                # convert from cv image to ndarray
                # input_image = np.array(input_image)
                # input_image = input_image.astype(np.float32)
                
                # apply transforms if passed
                if self.transform:
                        input_image = self.transform(input_image)

                # siphon hitch angle truth
                hitch_truth = self.df[self.column_names[9]][idx]
                hitch_truth = torch.as_tensor(hitch_truth) 

                return input_image, hitch_truth.float()

###################################################################################
#                              END OF FUNCTIONAL
#                               
#                         BEGINING OF TEST DRIVEN DEV
###################################################################################
