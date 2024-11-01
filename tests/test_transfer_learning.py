'''
############ Transfer Learning Test ############

            FOR TEST DRIVEN DEVELOPMENT

################################################
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.trainer import Trainer

#%%
# Set up modified network
model = torchvision.models.resnet34(weights='IMAGENET1K_V1')

# freeze layers for fixed-feature extraction
# for param in model.parameters():
#     param.requires_grad = False

# modify network head
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1)
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
torch.cuda.empty_cache()
model = model.to(device)

#%%
# Setup data

TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data.csv'
OUT_PATH = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\test_resnet34_full_weights.pth'
full_set = TrailerData(csv_file=TRAIN_CSV,
                       transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((512,512)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                ]))

# split training set to training/val sets
NUM_VAL = 1000
NUM_TRAIN = len(full_set) - NUM_VAL
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
# baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

BATCH_SIZE = 20
# generate loaders
loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
# loader_baby = DataLoader(baby_set, batch_size=BATCH_SIZE, shuffle=True)

#%%
# Setup Training

# optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=1e-4)

# define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

print("Training model ...")
print()

network_trainer = Trainer(model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_scale=100,
                          device=device,
                          verbose={"cond":True,"print_every":25},
                          save_weights={"cond": True, "save_path": OUT_PATH})

outs = network_trainer.train(epochs=10)

# plot loss
plt.subplot(3,1,1)
plt.plot(outs["loss_history"], '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plot accuracies
plt.subplot(3,1,2)
plt.plot(outs["rmse_train_history"], '-o')
plt.plot(outs["rmse_val_history"], '-o')
plt.legend(['train', 'val'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('RMSE [deg]')
plt.tight_layout()