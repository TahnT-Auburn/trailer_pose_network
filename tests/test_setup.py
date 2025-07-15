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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils

from trailer_pose_network.data_setup import TractorTrailerData
from trailer_pose_network.sequential_data_setup import SeqTrailerData
from trailer_pose_network.custom_transforms import *

#%% Generate and organize data tests
# NOTE: All code below is intended for test driven development.
#       Execution code is located outside this script
PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\simulation\\training\\"
FILE = "full_training.csv"
TRAIN_CSV = os.path.join(PARENT, FILE)

SEQ_PARENT = "D:\\TrainingData\\processed"
# TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data.csv'
output_states = ['hitch', 'hitch_rate']

full_set = TractorTrailerData(csv_file=TRAIN_CSV,
                                inputs={"cam":False, "can": True, "imu": True},
                                transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((512,512)),
                                    transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                    ]),
                                output_states=output_states,
                                sequential=4,
                                sequence_root=SEQ_PARENT,
                                )

# sequence_length = 3
# full_set = SeqTrailerData(csv_file=TRAIN_CSV,
#                         output_state=8,
#                         sequence_length=sequence_length,
#                         transform=transforms.Compose([
#                                  transforms.ToPILImage(),
#                                  transforms.Resize((224,224)),
#                                  transforms.ToTensor(),
#                                 #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
#                                 ]))

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

# print("Length of train set: %d" % loader_train.sampler.num_samples)
# print("Length of val set: %d" % loader_val.sampler.num_samples)
# print()
once = False
for i_batch, sample_data in enumerate(loader_val):
    if once:
        break
    inputs = sample_data[0]
    outputs = sample_data[1]
    print(f'\nNumber of inputs: {len(inputs)}')
    for input in inputs:
        print(f'Input shape: {input.shape}')
    print(f'Output: {outputs.shape}')
    # print(sample_data.shape)
    # B = sample_data.shape[0]
    # T = sample_data.shape[1]
    # sample_data = sample_data.squeeze()
    # for i in range(T):
    #     img = sample_data[i].permute(1,2,0)
    #     plt.imshow(img)
    #     plt.show()
    once = True


#         if once:
#                 break
#         image = sample_data[0]
#         # print(type(image))
#         out_states = sample_data[1]
#         len_batch = image.shape[0]
#         print(f"Input shape: {image.shape}")
#         print(f"Output Shape: {out_states.shape}")
        # for i in range(0,len_batch):
        #         plt.imshow(image[i].permute(1,2,0))
        #         # print(type(image))
        #         # plt.imshow(image[i])
        #         # print(grid.shape)
        #         print(f'Out States: {output_states} {out_states}')
        #         plt.show()
        # once = True


# for i_batch, sample_data in enumerate(loader_val):
#         image_sequence = np.array(sample_data[0])
#         target = sample_data[2]
#         input_image = sample_data[1]
#         len_batch = input_image.shape[0]
#         if i_batch == 0: # query one batch
#                 for i in range(0,len_batch):
#                         print("Target: %f" % target[i])
#                         for j in range(0,sequence_length):
#                                 plt.imshow(image_sequence[j][i].transpose(1,2,0))
#                                 plt.show()
#                 break

# %%
