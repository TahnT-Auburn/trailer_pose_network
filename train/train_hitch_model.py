#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.dataloaders.trailer_hitch_dataloader import HitchDataloader
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel

from trailer_pose_network.trainers.trainer_hitch_model import Trainer

#%%
# Set Global variables

# === FILE LOADING ===
TRAIN_CSV_ROOT = "D:\\TrainingData\\simulation\\processed\\"

VAL_CSV_ROOT = "D:\\TestingData\\simulation\\processed\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\trailer_hitch"
WEIGHT_FILE = "sim_v1.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)
SAVE_WEIGHTS = WEIGHT_SAVE_PATH

SAVE_LOG = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\logs\\trailer_hitch\\sim_v1\\training_log.csv"

PRETRAINED_WEIGHTS = None
PRETRAINED = False

# === DATALOADER PARAMETERS ===
IMG_SIZE = (224,224)
BATCH_SIZE = 6
NUM_WORKERS = 4

# === MODEL PARAMETERS ===
ENCODER = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')
EMBED_DIM = 784
DROPOUT = 0.

# === TRAINING PARAMETERS ===
NUM_EPOCHS = 30
LR = 1e-4
LOSS_SCALE = 1e3
LOSS_FUNC = nn.MSELoss()
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.05
WARMUP_PERIOD = 2 # The number of epochs to warmup
OVERFIT_DETECTOR = False

#%%
def train():
    # Load dataset
    train_set = HitchDataloader(
        csv_root=TRAIN_CSV_ROOT,
        # reduce={'target_column':'steer_ang', 'target_size':10},
        transforms=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToTensor(),
        ]),
    )
    val_set = HitchDataloader(
        csv_root=VAL_CSV_ROOT,
        reduce={'target_column':'hitch', 'target_size':1500},
        transforms=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToTensor(),
        ]),
    )
    # num_val = int(np.round(VAL_RATIO * len(full_set)))
    # num_train = len(full_set) - num_val
    # train_set, val_set = random_split(full_set, [num_train, num_val])
    
    # Generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # Load model
    model = HitchModel(
        encoder=ENCODER,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device is Use: %s' % device)
    model = model.to(device)
    
    # Load pretrained weights if prompted
    if PRETRAINED:
        state_dict = torch.load(PRETRAINED_WEIGHTS)
        model.load_state_dict(state_dict)
        
    # Set up training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=(NUM_EPOCHS)*len(loader_train), eta_min=0.0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*2, T_mult=2)
    warmup_period = len(loader_train) * WARMUP_PERIOD
    warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)
    
    network_trainer = Trainer(model=model,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              warmup_scheduler=warmup_scheduler,
                              loader_train=loader_train,
                              loader_val=loader_val,
                              overfit_detector=OVERFIT_DETECTOR,
                              loss_scale=LOSS_SCALE,
                              device=device,
                              verbose=len(loader_train),
                              save_weights=SAVE_WEIGHTS,
                              save_outs=SAVE_LOG)
    
    # Train model
    print('Training ...')
    print ()
    
    start_time = time.time()
    model, outs = network_trainer.train(loss_func=LOSS_FUNC, epochs=NUM_EPOCHS)
    print("Total training time: %s" % (time.time() - start_time))
    print()
    print("Training Complete")
    print()
    
    return model, outs

#%%
# Call training and visualize training
if __name__ == "__main__":
    
    model, outs = train()
    
    # plot loss
    plt.subplot(4,1,1)
    plt.plot(outs["lr_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.subplot(4,1,2)
    plt.plot(outs["train_loss_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.subplot(4,1,3)
    plt.plot(outs["val_loss_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Validation Loss')
    plt.subplot(4,1,4)
    plt.plot(outs["train_epoch_loss_history"], '-o')
    plt.plot(outs["val_epoch_loss_history"])
    plt.legend(["Train", "Val"])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

    # plot accuracies
    L = len(outs["rmse_train_history"])
    # train_rmse = np.concatenate(outs["rmse_train_history"]).reshape(L,NUM_OUTPUTS)
    # val_rmse = np.concatenate(outs["rmse_val_history"]).reshape(L,NUM_OUTPUTS)
    train_rmse = outs["rmse_train_history"]
    val_rmse = outs["rmse_val_history"]
    # Loop truth and plot each output as a subplot
    plt.plot(train_rmse, '-o')
    plt.plot(val_rmse, '-o')
    plt.legend(['Train', 'Val'])
    plt.ylabel('OUTPUT RMSE')
    plt.xlabel('Epochs')
            