#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.visual_odometry.raft_based_visual_odom_model import RaftBasedVisualOdom
from trailer_pose_network.trainers.trainer_resnet_based_visual_odom_model import Trainer
#%%
# set global variables
SEQ_ROOT_PROCESSED = "D:\\TrainingData\\experimental\\10Hz\\original\\"
SEQ_ROOT_RAW = "D:\\TrainingData\\experimental\\40Hz\\original\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\visual_odometry\\"
WEIGHT_FILE = "raft_based_visual_odom_v1.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)
SAVE_WEIGHTS = WEIGHT_SAVE_PATH

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 6
VAL_RATIO = 0.2
NUM_WORKERS = 4

# === MODEL PARAMETERS ===
EMBED_DIM = 384

# === TRAINING PARAMETERS ===
NUM_EPOCHS = 15
LR = 1e-4
LOSS_SCALE = 1e1
LOSS_FUNC = nn.MSELoss()
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.05
WARMUP_PERIOD = 2 # The number of epochs to warmup
RUN_VAL = True
OVERFIT_DETECTOR = False
CHECK_ACCURACY = True
CHECK_GRADIENTS = False

def train():
    full_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        sequential_lookback=NUM_FRAMES,
        inputs={'cam':True, 'can':False, 'imu':False, 'yaw_hist':False},
        reduce={'target_column':'steer_ang', 'target_size':5000},
        transform_img=T.Compose([
                T.ToPILImage(),
                T.Resize(IMG_SIZE),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),
            ]),
        )
    num_val = int(np.round(VAL_RATIO * len(full_set)))
    num_train = len(full_set) - num_val
    train_set, val_set = random_split(full_set, [num_train, num_val])
    
    # Generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    model = RaftBasedVisualOdom(
        embed_dim=EMBED_DIM
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*2, T_mult=2)
    warmup_period = len(loader_train) * WARMUP_PERIOD
    warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)
    
    network_trainer = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler,
                            loader_train=loader_train,
                            loader_val=loader_val,
                            run_val=RUN_VAL,
                            overfit_detector=OVERFIT_DETECTOR,
                            loss_scale=LOSS_SCALE,
                            loss_save_interval=None,
                            device=device,
                            check_accuracy=CHECK_ACCURACY,
                            check_gradients=CHECK_GRADIENTS,
                            verbose=len(loader_train),
                            save_weights=SAVE_WEIGHTS)
    
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

if __name__ == '__main__':
    model, outs = train()
    
    if RUN_VAL:
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
    else:
        plt.subplot(3,1,1)
        plt.plot(outs["lr_history"])
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.subplot(3,1,2)
        plt.plot(outs["train_loss_history"])
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.subplot(3,1,3)
        plt.plot(outs["train_epoch_loss_history"], '-o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()
    
    # plot accuracies
    # plt.plot(outs["rmse_train_history"], '-o')
    # if RUN_VAL:
    #     plt.plot(outs["rmse_val_history"], '-o')
    #     plt.legend(['Train', 'Val'])
    #     plt.ylabel('RMSE')
    #     plt.xlabel('Epochs')
    
    NUM_OUTPUTS = 2
    if CHECK_ACCURACY:
        L = len(outs["rmse_train_history"])
        train_rmse = np.concatenate(outs["rmse_train_history"]).reshape(L,NUM_OUTPUTS)
        if RUN_VAL: 
            val_rmse = np.concatenate(outs["rmse_val_history"]).reshape(L,NUM_OUTPUTS)
            
    # Loop truth and plot each output as a subplot
    for i in range(NUM_OUTPUTS):
        plt.subplot(NUM_OUTPUTS,1,i+1)
        plt.plot(train_rmse[:,i], '-o')
        if RUN_VAL:
            plt.plot(val_rmse[:,i], '-o')
            plt.legend(['Train', 'Val'])
        state_num = str(i+1)
        plt.ylabel('OUTPUT' + state_num + ' RMSE')
        plt.xlabel('Epochs')