'''
################ Data Setup Testing Script ################
#
#               FOR TEST DRIVEN DEVELOPEMNT
#
###########################################################
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

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.custom_transforms import *

#%% Generate and organize data tests
# NOTE: All code below is intended for test driven development.
#       Execution code is located outside this script

TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data.csv'
full_set = TrailerData(csv_file=TRAIN_CSV,
                       transform=transforms.Compose([
                                 Rescale((512,512)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                ]))

# transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
#  transforms.Normalize((0.2983605, 0.30982107, 0.3277596 ),(0.3057412, 0.3069248, 0.31205365)),

# full_set = TrailerData(csv_file=TRAIN_CSV,
#                        transform=None)

# split training set to training/val sets
NUM_VAL = 1000
NUM_TRAIN = len(full_set) - NUM_VAL
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])

# generate loaders
loader_train = DataLoader(train_set, batch_size=2, shuffle=True)
loader_val = DataLoader(val_set, batch_size=2, shuffle=True)

# # find mean and std of full training set
# loader_full = DataLoader(full_set, batch_size=1024, shuffle=True)
# images, _ = next(iter(loader_full))
# numpy_images = images.numpy()
# print(numpy_images.shape)

# channel_mean = np.mean(numpy_images, axis=(0,2,3))
# channel_std = np.std(numpy_images, axis=(0,2,3))

# print("Channel mean:")
# print(channel_mean)
# print("Channel STD:")
# print(channel_std)

#%% Sanity check plotting

print("Length of train set: %d" % loader_train.sampler.num_samples)
print("Length of val set: %d" % loader_val.sampler.num_samples)
print()

for i_batch, sample_data in enumerate(loader_val):
        image = sample_data[0]
        hitch = sample_data[1]
        len_batch = image.shape[0]
        if i_batch == 0: # query once
                for i in range(0,len_batch):
                        plt.imshow(image[i].permute(1,2,0))
                        # print(type(image))
                        # plt.imshow(image[i])
                        # print(grid.shape)
                        print('Hitch angle: %.2f' % np.rad2deg(hitch[i]))
                        plt.show()