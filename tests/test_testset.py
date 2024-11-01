'''
################  Test Set Testing Script ################
#
#               FOR TEST DRIVEN DEVELOPEMNT
#
##########################################################
'''
#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from trailer_pose_network.custom_transforms import *
from trailer_pose_network.models.mangonet import MangoNet
from trailer_pose_network.models.vanillanet1 import VanillaNet1
from trailer_pose_network.data_setup import TrailerData

#%%
# load model with trained weights

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set model params
model_params={
        "shape_in": (3,512,512), 
        "initial_filters": 32,    
        "num_fc1": 300,
        "dropout_rate": 0.0}

# model_params={
#         "shape_in": (3,512,512), 
#         "initial_filters": 16,    
#         "num_fc1": 500,
#         "num_fc2": 300,
#         "dropout_rate": 0.0}

# set weight file
weight_file = "test_resnet34_full_weights.pth"

# load model(s)
# model = MangoNet(model_params)
# model = VanillaNet1(model_params)

# Set up modified network
model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
num_features = model.fc.in_features # modify network head
model.fc = nn.Sequential(
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1)
)

model = model.to(device)
weight_path = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\" + weight_file
state_dict = torch.load(weight_path)
model.load_state_dict(state_dict)

#%%
# load data using pytorch dataset and dataloader tools
# TEST_CSV = "/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/processed/INT/INT2/INT2_training.csv"

TEST_CSV = "D:\\TestingData\\Processed\\HWY\\HWY1\\HWY1_testing.csv"
# TEST_CSV = "D:\\TestingData\\processed\\INT\\INT1\\INT1_testing.csv"
test_set = TrailerData(csv_file=TEST_CSV,
                       transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((512,512)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                ]))

# generate loader
loader_test = DataLoader(test_set, batch_size=1, shuffle=False)

#%%
# evalulate model
est_array = []
truth_array = []
with torch.no_grad():
    model.eval()
    for t, (x,y) in enumerate(loader_test):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        est = model(x)

        est_array.append(est.item())
        truth_array.append(y.item())

#%% plot estimates vs truth
plt.subplot(2,1,1)
plt.plot(np.rad2deg(est_array))
plt.plot(np.rad2deg(truth_array))
plt.xlabel('Frames')
plt.ylabel('Hitch [deg]')
plt.legend(['Estimated', 'Truth'], loc='upper right')
plt.subplot(2,1,2)
plt.plot(np.rad2deg(truth_array) - np.rad2deg(est_array))
plt.xlabel('Frames')
plt.ylabel('Error [deg]')
plt.tight_layout()
plt.show()