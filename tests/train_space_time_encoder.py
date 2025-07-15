"""
Space-Time Vision Transformer Testing Script
"""
#%%
import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.data_setup import TractorTrailerData
from trailer_pose_network.models.space_time_encoder import SpaceTimeEncoder
from trailer_pose_network.models.space_time_early_fusion import SpaceTimeEncoderEarlyFusion

from trailer_pose_network.trainer import Trainer
from sklearn.model_selection import train_test_split

#%%
# Generate dataloaders
IN_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\simulation\\training\\"
IN_FILE = "full_training.csv"
TRAIN_CSV = os.path.join(IN_PARENT, IN_FILE)
SEQ_PARENT = "D:\\TrainingData\\simulation\\processed"

WEIGHT_PARENT = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\tests\\"
WEIGHT_FILE = "space_time.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

num_frames = 2
img_size = 224
full_set = TractorTrailerData(csv_file=TRAIN_CSV,
                                inputs={"cam":False, "can": True, "imu": True},
                                reduce={"target_column":"hitch", "target_size":10000},
                                transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((img_size,img_size)),
                                    transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                    ]),
                                sequential=num_frames,
                                sequence_root=SEQ_PARENT,
                                )

# split training set to training/val sets
NUM_VAL = int(np.round(0.1*len(full_set)))
NUM_TRAIN = len(full_set) - NUM_VAL
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
# tiny_set,val_set, remainder = random_split(full_set,[0.1,0.1,0.8])

# generate loaders
loader_train = DataLoader(train_set, batch_size=6, shuffle=True)
loader_val = DataLoader(val_set, batch_size=6, shuffle=True)
# loader_tiny = DataLoader(tiny_set, batch_size=6, shuffle=True)

#%%
# Load model
# model = SpaceTimeEncoder(num_frames=num_frames,
#                         inp_size=(img_size,img_size),
#                         patch_size=16,
#                         in_channels=3,
#                         embed_dim=384)

model = SpaceTimeEncoder(num_frames=num_frames,
                        inp_size=(1,4),
                        patch_size=1,
                        in_channels=1,
                        embed_dim=384)

# model = SpaceTimeEncoderEarlyFusion(num_frames=num_frames,
#                                     embed_dim=384,
#                                     img_size=(img_size,img_size), img_patch_size=16, img_channels=3,
#                                     inp_size=(1,4), inp_patch_size=1, inp_channels=1,
#                                     num_heads=8, depth=12,
#                                     attn_drop=0.2, proj_drop=0.2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
model = model.to(device)

#%%
# Setup training
num_epochs = 10
num_iters = len(loader_train) * num_epochs
warmup_period = len(loader_train)//2 # half an epoch

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*5, T_mult=1)
warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)

print("Training ...")
print()

network_trainer = Trainer(model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          warmup_scheduler=warmup_scheduler,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_scale=1e1,
                          device=device,
                          check_accuracy=True,
                          verbose=len(loader_train),
                          save_weights=WEIGHT_SAVE_PATH)

start_time = time.time()
outs = network_trainer.train(loss_func=nn.MSELoss(), epochs=num_epochs)
print("Training Time: %s" % (time.time() - start_time))
print()

# plot loss
plt.subplot(3,1,1)
plt.plot(outs["lr_history"])
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.subplot(3,1,2)
plt.plot(outs["raw_loss_history"])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.subplot(3,1,3)
plt.plot(outs["loss_history"], '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()

# plot accuracies
L = len(outs["rmse_train_history"])
train_rmse = np.concatenate(outs["rmse_train_history"]).reshape(L,2)
val_rmse = np.concatenate(outs["rmse_val_history"]).reshape(L,2)
hitch_rmse_train = train_rmse[:,0]
hitch_rmse_val = val_rmse[:,0]
hr_rmse_train = train_rmse[:,1]
hr_rmse_val = val_rmse[:,1]
plt.subplot(2,1,1)
plt.plot(hitch_rmse_train, '-o')
plt.plot(hitch_rmse_val, '-o')
plt.legend(['train', 'val'], loc='upper right')
plt.ylabel('Hitch RMSE [deg]')
plt.xlabel('Epochs')
plt.subplot(2,1,2)
plt.plot(hr_rmse_train, '-o')
plt.plot(hr_rmse_val, '-o')
plt.ylabel('Hitch Rate RMSE [deg/s]')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

#%%
# # test a single instance
# SINGLE_CSV = "D:\\TestingData\\processed\\FF\\FF2\\FF2.csv"
# WEIGHTS = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\tests\\space_time.pth"
# # data loader
# test_set = TractorTrailerData(csv_file=SINGLE_CSV,
#                               sequential=num_frames,
#                               sequence_root=SEQ_PARENT,
#                               transform=transforms.Compose([
#                                     transforms.ToPILImage(),
#                                     transforms.Resize((img_size,img_size)),
#                                     transforms.ToTensor(),
#                                 ]))
# test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# # model
# # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = SpaceTimeEncoder(num_frames=num_frames,
#                         inp_size=(img_size,img_size),
#                         patch_size=16,
#                         in_channels=3,
#                         embed_dim=384)

# model = model.to(device)
# state_dict = torch.load(WEIGHTS)
# model.load_state_dict(state_dict)

# # evalulate single model
# est_array = []
# truth_array = []
# print("Evaluating Model ...")
# with torch.no_grad():
#     model.eval()
#     for t, (x,y) in enumerate(test_loader):
#         start_time = time.time()
#         if isinstance(x,list):
#             x = x[0] # grab first TODO: Modify this to be more interactive. Make num_inputs a parameter
#         x = x.to(device=device, dtype=torch.float32)
#         y = y.to(device=device, dtype=torch.float32)

#         est = model(x)

#         est_array.append(est)
#         truth_array.append(y)
        
#         print(f"Iter: {t}")
#         # print(f"Single loop time: {time.time()-start_time}")
#         stop=1
# print("Evaluation Complete")
# %%
