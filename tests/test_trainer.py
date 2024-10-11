'''
################ Trainer Class Testing Script ################
#
#                 FOR TEST DRIVEN DEVELOPEMNT
#
##############################################################
'''
#%%
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from trailer_pose_network.custom_transforms import *
from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.mangonet import MangoNet
from trailer_pose_network.trainer import Trainer

#%%
# Generate and organize data 
print("Setting up data ...")

TRAIN_CSV = '/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/reduced/reduced_data.csv'
full_set = TrailerData(csv_file=TRAIN_CSV,
                       transform=transforms.Compose([
                                 Rescale((512,512)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
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
# baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

# generate loaders
loader_train = DataLoader(train_set, batch_size=32, shuffle=True)
loader_val = DataLoader(val_set, batch_size=32, shuffle=True)
# loader_baby = DataLoader(baby_set, batch_size=32, shuffle=True)

print("Length of train set: %d" % loader_train.sampler.num_samples)
print("Length of val set: %d" % loader_val.sampler.num_samples)
print()

#%%
# Setup training
model_params={
        "shape_in": (3,512,512), 
        "initial_filters": 32,    
        "num_fc1": 200,
        "dropout_rate": 0.5}

model = MangoNet(model_params)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
torch.cuda.empty_cache()
# CUDA_LAUCH_BLOCKING=1
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-7, weight_decay=1e-4)
print("Training model ...")
print()

network_trainer = Trainer(model=model,
                          optimizer=optimizer,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_scale=60,
                          device=device,
                          verbose={"cond":True,"print_every":25})

loss_history, mse_train_history, mse_val_history = network_trainer.train(epochs=5)

# plot loss
plt.subplot(2,1,1)
plt.plot(loss_history, '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plot accuracies
plt.subplot(2,1,2)
plt.plot(mse_train_history, '-o')
plt.plot(mse_val_history, '-o')
plt.legend(['train', 'val'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('MSE [deg]')
plt.tight_layout()
plt.show()