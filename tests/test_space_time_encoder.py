#%%
import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
import time
from time import sleep
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as Fs
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.data_setup import TractorTrailerData
from trailer_pose_network.models.space_time_encoder import SpaceTimeEncoder
from trailer_pose_network.models.space_time_early_fusion import SpaceTimeEncoderEarlyFusion
from trailer_pose_network.trainer import Trainer
from sklearn.model_selection import train_test_split

SEQ_PARENT = "D:\\TrainingData\\simulation\\processed\\INT\\INT1"
SINGLE_CSV = "D:\\TrainingData\\simulation\\processed\\INT\\INT1\\INT1.csv"
# SEQ_PARENT = "D:\\TestingData\\simulation\\processed\\FF\\FF2"
# SINGLE_CSV = "D:\\TestingData\\simulation\\processed\\FF\\FF2\\FF2.csv"
WEIGHTS = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\tests\\space_time.pth"

# data loader
num_frames = 2
img_size = 224

test_set = TractorTrailerData(csv_file=SINGLE_CSV,
                              inputs={"cam":False, "can": True, "imu": True},
                              single_test=True,
                              sequential=num_frames,
                              sequence_root=SEQ_PARENT,
                              transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                ]))
BATCH_SIZE = 24
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
L = len(test_set)

# model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SpaceTimeEncoder(num_frames=num_frames,
                            inp_size=(1,4),
                            patch_size=1,
                            in_channels=1,
                            embed_dim=384)

# model = SpaceTimeEncoderEarlyFusion(num_frames=num_frames,
#                                     embed_dim=384,
#                                     img_size=(img_size,img_size), img_patch_size=16, img_channels=3,
#                                     inp_size=(1,4), inp_patch_size=1, inp_channels=1,
#                                     num_heads=8, depth=12)
model = model.to(device)
state_dict = torch.load(WEIGHTS)
model.load_state_dict(state_dict)

# evalulate single model
est_array = []
truth_array = []
print("Evaluating Model ...")
with torch.no_grad():
    model.eval()
    for t, (x,y) in enumerate(tqdm(test_loader)):
        # sleep(0.01)
        # start_time = time.time()
        if isinstance(x,list):
            if len(x) == 2:
                x[0] = x[0].to(device=device, dtype=torch.float32)
                x[1] = x[1].to(device=device, dtype=torch.float32)
            else:
                x = x[0] # grab first TODO: Modify this to be more interactive. Make num_inputs a parameter
                x = x.to(device=device, dtype=torch.float32)
        else:
            x = x.to(device=device, dtype=torch.float32)

        y = y.to(device=device, dtype=torch.float32)

        est = model(x)

        est_array.append(est)
        truth_array.append(y)
        
        # print(f"Percent Complete: {(t*BATCH_SIZE/L)*100}")
        # print(f"Single loop time: {time.time()-start_time}")
        stop=1
est_array = torch.cat(est_array).cpu().numpy()
truth_array = torch.cat(truth_array).cpu().numpy()

print("Evaluation Complete")
# %%
# plot
# hitch_est = np.rad2deg(est_array[:,0])
# hr_est = np.rad2deg(est_array[:,1])
# hitch_truth = np.rad2deg(truth_array[:,0])
# hr_truth = np.rad2deg(truth_array[:,1])
hitch_est = est_array[:,0]
hr_est = est_array[:,1]
hitch_truth = truth_array[:,0]
hr_truth = truth_array[:,1]

plt.subplot(211)
plt.plot(hitch_truth,'r')
plt.plot(hitch_est,'k')
plt.legend(["Truth", "Est"])
plt.xlabel('Frames')
plt.ylabel('Hitch [deg]')
plt.subplot(212)
plt.plot(hitch_truth - hitch_est, 'k')
plt.xlabel('Frames')
plt.ylabel('Error [deg]')
plt.tight_layout()
plt.show()

plt.subplot(211)
plt.plot(hr_truth,'r')
plt.plot(hr_est,'k')
plt.legend(["Truth", "Est"])
plt.xlabel('Frames')
plt.ylabel('Hitch Rate [deg/s]')
plt.subplot(212)
plt.plot(hr_truth - hr_est, 'k')
plt.xlabel('Frames')
plt.ylabel('Error [deg/s]')
plt.tight_layout()
plt.show()
# %%
