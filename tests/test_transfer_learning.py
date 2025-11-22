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
import time
import pandas as pd

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.trainer import Trainer

NET = 'mobilenetv2'
TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\experimental\\reduced\\combined_set_v1_5k.csv'
OUT_PATH = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\experimental\\combined_v1\\'+NET+'_weights_5k.pth'
TRAIN_SPECS = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\experimental\\combined_v1\\'+ NET +'_train_specs_5k.csv'
# OPT = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\optuna_narrow\\'+ NET +'_optuna_studies_narrow.csv'

#%%
# Set up modified network
# 'IMAGENET1K_V1'
model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
model = torchvision.models.resnet34(weights=None)

# torchvision.models.resnet18
output_states = 2
# torchvision.models.mobilenet_v2
#TODO: Make custom network for alexnet

# freeze layers for fixed-feature extraction
# for param in model.parameters():
#     param.requires_grad = False

# modify network head
# num_features = model.fc.in_features
num_features = model.classifier[1].in_features

# model.classifier = nn.Sequential(
#     nn.Linear(num_features, num_features*2),
#     nn.ReLU(),
#     nn.Linear(num_features*2, num_features//2),
#     nn.ReLU(),
#     nn.Linear(num_features//2, num_features//4),
#     nn.ReLU(),
#     nn.Linear(num_features//4, 1)
# )

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

model.classifier = nn.Sequential(             # standard head
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1),
)

# model.classifier = nn.Sequential(     # samll head
#     nn.Linear(num_features, 1)
# )

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
torch.cuda.empty_cache()
model = model.to(device)

undistort = {
    "cameraMatrices":
    [np.array([[1209.487477, 0.000000, 1047.120765],
                        [0.000000, 1210.806093, 766.413057],
                        [0.000000, 0.000000, 1.000000]]),
    np.array([[1180.212713, 0.000000, 1009.890502],
                        [0.000000, 1182.408953, 758.997736],
                        [0.000000, 0.000000, 1.000000]])
    ],
    "distCoeffs":
    [np.array([-0.225053, 0.067380, -0.000443, -0.000453, 0.000000]),
        np.array([-0.234383, 0.075105, -0.000760, 0.000797, 0.000000])]
    
}

#%%
##### visualize first filters #####

# grab first conv filter 
k = None
once = False
for m in model.modules():
    if not once:
        if isinstance(m, nn.Conv2d):
            k = m
            print(k)
            filter = k.weight.data.cpu()
            once = True

# visulizer function
def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


visTensor(filter, ch=0, allkernels=False)
plt.axis('off')
plt.ioff()
plt.show()

#%%
# Setup data
full_set = TrailerData(csv_file=TRAIN_CSV,
                       output_states_idx=output_states,
                       transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((512,512)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                ]),
                        undistort=undistort
                                )

# split training set to training/val sets
train_df = pd.read_csv(TRAIN_CSV)
# NUM_VAL = 1000
NUM_VAL = int(0.1*len(train_df)) # make validation set 10% of training set
NUM_TRAIN = int(len(full_set) - NUM_VAL)
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
# baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

BATCH_SIZE = 6
# generate loaders
loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
# loader_baby = DataLoader(baby_set, batch_size=BATCH_SIZE, shuffle=True)

#%%
# Setup Training

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

print("Training " + NET + " ...")
print()

network_trainer = Trainer(model=model,
                          optimizer=optimizer,
                          scheduler=None,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_scale=1e3,
                          device=device,
                          verbose=100,
                          save_weights=OUT_PATH,
                          save_outs=TRAIN_SPECS)

start_time = time.time()
outs = network_trainer.train(epochs=4)
print("Training Time: %s" % (time.time() - start_time))

# plot loss
plt.subplot(3,1,1)
plt.plot(outs["raw_loss_history"])
plt.subplot(3,1,2)
plt.plot(outs["loss_history"], '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plot accuracies
plt.subplot(3,1,3)
plt.plot(outs["rmse_train_history"], '-o')
plt.plot(outs["rmse_val_history"], '-o')
plt.legend(['train', 'val'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('RMSE [deg]')
plt.tight_layout()
plt.show()

# # alternative plot (reports loss only after each epoch)
# plt.subplot(3,1,1)
# plt.plot(outs["loss_history"][1:], '-')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# # plot accuracies
# plt.subplot(3,1,2)
# plt.plot(outs["rmse_train_history"], '-o')
# plt.plot(outs["rmse_val_history"], '-o')
# plt.legend(['train', 'val'], loc='upper right')
# plt.xlabel('Epochs')
# plt.ylabel('RMSE [deg]')
# plt.tight_layout()
# plt.show()