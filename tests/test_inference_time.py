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
import time

from trailer_pose_network.custom_transforms import *
from trailer_pose_network.data_setup import TrailerData

NET = "densenet121"
TEST_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\combined\\combined_testing_data.csv"
WEIGHTS = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\plans\\ablations\\network_head\\small\\"+NET+"_weights.pth"

#%%
# load model with trained weights

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set up modified network
model = torchvision.models.densenet121(weights=False)
# num_features = model.fc.in_features # modify network head
num_features = model.classifier.in_features

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

model.classifier = nn.Sequential(     # samll head
    nn.Linear(num_features, 1)
)

# model.classifier = nn.Sequential(
#     nn.Linear(num_features, 500),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(500,300),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(300,1)
# )

model = model.to(device)
state_dict = torch.load(WEIGHTS)
model.load_state_dict(state_dict)
# model.load_state_dict(state_dict, strict=False)

#%% Load a single image
test_set = TrailerData(csv_file=TEST_CSV,
                       transform=transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.Resize((512,512)),
                           transforms.ToTensor(),
                       ]))
BATCHSIZE = 1
loader_test = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True)

inf_times = []
img_pool = 1000
print("Evaluating ...")
with torch.no_grad():
    model.eval()
    warmed_up = False
    for t, (x,y) in enumerate(loader_test):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)
        if t == img_pool:
            break
        # print(t)
        # Warm-up runs
        if not warmed_up:
            for _ in range(10):
                _ = model(x)
                warmed_up = True
        start_time = time.time()
        est = model(x)
        torch.cuda.synchronize()
        inf_time = time.time() - start_time
        inf_times.append(inf_time)
        # print("Training Time: %s" % (time.time() - start_time))
# eliminate first entry
avg_inf_time = np.mean(inf_times[1:])
print("Average inference time for %s over %d images: %f" % (NET, img_pool,avg_inf_time))
# %%
