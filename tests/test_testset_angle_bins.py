'''
################  Test Set Testing Script ################
#
#               FOR TEST DRIVEN DEVELOPEMENT
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
import matplotlib.animation as animation
import os

from trailer_pose_network.custom_transforms import *
from trailer_pose_network.models.mangonet import MangoNet
from trailer_pose_network.models.vanillanet1 import VanillaNet1
from trailer_pose_network.data_setup import TrailerData

#%% 
# set weight file
NET = "resnet34"
BIN_CSV_PARENT= "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\binned"
WEIGHTS = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\plans\\finetune_transfer_learning\\reducedv2\\"+NET+"_weights.pth"
OUT_CSV_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\angle_bin_results\\reducedv2"
#%%
# helper functions
# Compute RMSE of testing set
def rmse(x_true, x_pred):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return np.sqrt(np.mean((x_true - x_pred)**2))


def genErrors(x_true, x_pred):
    '''
    calculates the error between an x_truth and x_pred (values or arrays)
    '''
    return x_true - x_pred

#%%
# load model with trained weights

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set up modified network
model = torchvision.models.resnet34(weights=False)
num_features = model.fc.in_features # modify network head
# num_features = model.classifier[1].in_features

# model.classifier = nn.Sequential(             # large head
#     nn.Linear(num_features, 1500),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(1500,750),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(750,375),
#     nn.ReLU(),
#     nn.Linear(375,1)
# )

# model.classifier = nn.Sequential(     # samll head
#     nn.Linear(num_features, 1)
# )

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
state_dict = torch.load(WEIGHTS)
model.load_state_dict(state_dict)
# model.load_state_dict(state_dict, strict=False)

#%%
# Evaluate each bin dataset and group by network

# loop through bin folder and load individual csvs
for root,_,files in os.walk(BIN_CSV_PARENT):
    # find and evalulate indivual bin files
    for idx, file in enumerate(files):
        # if first pass, generate an empty .csv file to populate
        if idx == 0:
            df = pd.DataFrame()
            df.to_csv(OUT_CSV_PARENT+'\\'+NET+'.csv')

        TEST_CSV = root+'\\'+file
        print(f"Evaluating {TEST_CSV}")

        test_set = TrailerData(csv_file=TEST_CSV,
                               output_states=9,
                                transform=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                        #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                        #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                        ]))

        # generate loader
        BATCHSIZE = 20
        loader_test = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True)

        # evalulate model
        est_array = []
        truth_array = []
        with torch.no_grad():
            model.eval()
            for t, (x,y) in enumerate(loader_test):
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                est = model(x)

                est_array.extend(est.squeeze().cpu().numpy())
                truth_array.extend(y.cpu().numpy())
        

        # calculate rmse of current split
        rmse_test = rmse(np.array(truth_array), np.array(est_array))
        rmse_test = np.rad2deg(rmse_test)

        # calculate error
        error = genErrors(np.array(truth_array), np.array(est_array))
        error = list(np.rad2deg(error))

        # populate a new column w/ name respect to the current bin tested
        df[os.path.splitext(file)[0]] = error
        df.to_csv(OUT_CSV_PARENT+'\\'+NET+'.csv', index=False)

print("Evaluation Complete")

# rmse_test = rmse(np.array(truth_array), np.array(est_array))
# rmse_test = np.rad2deg(rmse_test)
# print("Test Set RMSE: %f" % rmse_test)




