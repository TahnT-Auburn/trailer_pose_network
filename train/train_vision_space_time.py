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

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.visual_odometry.visual_self_attn import ResNetBasedSelfAttention

# from trailer_pose_network.trainers.trainer_acc_yaw_deltas import Trainer
# from trailer_pose_network.trainer import Trainer
from trailer_pose_network.trainers.trainer_resnet_based_visual_odom_model import Trainer

#%%
# Set Global variables

# === FILE LOADING ===
# SEQ_ROOT_PROCESSED = "D:\\TrainingData\\experimental\\10Hz\\original\\"
# SEQ_ROOT_RAW = "D:\\TrainingData\\experimental\\40Hz\\original\\"

# SEQ_ROOT_PROCESSED_VAL = "D:\\TestingData\\experimental\\10Hz\\original\\"
# SEQ_ROOT_RAW_VAL = "D:\\TestingData\\experimental\\40Hz\\original\\"

SEQ_ROOT_PROCESSED = "D:\\TrainingData\\simulation\\10Hz\\"
SEQ_ROOT_RAW = "D:\\TrainingData\\simulation\\processed\\"

SEQ_ROOT_PROCESSED_VAL = "D:\\TestingData\\simulation\\10Hz\\"
SEQ_ROOT_RAW_VAL = "D:\\TestingData\\simulation\\processed\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\vision_space_time"
WEIGHT_FILE = "sim_v0.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)
SAVE_WEIGHTS = WEIGHT_SAVE_PATH

SAVE_LOG = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\logs\\space_time\\vision_space_time\\sim_v0\\training_log.csv"

PRETRAINED_WEIGHTS = None
PRETRAINED = False

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 4
VAL_RATIO = 0.2
NUM_WORKERS = 4

# === MODEL PARAMETERS ===
NUM_FRAMES = 2
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
PATCH_SIZE = 16
IN_CHANNELS = 3
DROPOUT = 0.
NUM_OUTPUTS = 3

# === TRAINING PARAMETERS ===
NUM_EPOCHS = 30
LR = 1e-4
LOSS_SCALE = 3e2
LOSS_FUNC = nn.MSELoss()
# LOSS_SCALE = 1e1
# LOSS_FUNC = nn.L1Loss()
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.05
WARMUP_PERIOD = 2 # The number of epochs to warmup
RUN_VAL = True
OVERFIT_DETECTOR = False
CHECK_ACCURACY = True
CHECK_GRADIENTS = False

#%%
def train():
    # Load dataset
    train_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':False, 'imu':False, 'yaw_hist':False},
        # reduce={'target_column':'steer_ang', 'target_size':5},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToTensor(),
        ]),
        augment_imu=True,
    )
    val_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED_VAL,
        sequence_root_raw=SEQ_ROOT_RAW_VAL,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':False, 'imu':False, 'yaw_hist':False},
        reduce={'target_column':'steer_ang', 'target_size':1500},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToTensor(),
        ]),
        augment_imu=True,
    )
    # num_val = int(np.round(VAL_RATIO * len(full_set)))
    # num_train = len(full_set) - num_val
    # train_set, val_set = random_split(full_set, [num_train, num_val])
    
    # Generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # Load model
    model = ResNetBasedSelfAttention(
        resnet_model=torchvision.models.resnet34(weights='IMAGENET1K_V1'),
        embed_dim=EMBED_DIM,
        num_frames=NUM_FRAMES,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        dropout=DROPOUT
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
                              run_val=RUN_VAL,
                              overfit_detector=OVERFIT_DETECTOR,
                              loss_scale=LOSS_SCALE,
                              loss_save_interval=None,
                              device=device,
                              check_accuracy=CHECK_ACCURACY,
                              check_gradients=CHECK_GRADIENTS,
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
    
    if RUN_VAL:
        # plot loss
        plt.subplot(4,1,1)
        plt.plot(outs["lr_history"])
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.subplot(4,1,2)
        plt.plot(outs["train_total_loss_history"])
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.subplot(4,1,3)
        plt.plot(outs["val_total_loss_history"])
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
        plt.plot(outs["train_total_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.subplot(3,1,3)
        plt.plot(outs["train_epoch_loss_history"], '-o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

    # if RUN_VAL:
    #     plt.subplot(411)
    #     plt.plot(outs["train_trans_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Train Loss1 (Trans)')
    #     plt.subplot(412)
    #     plt.plot(outs["train_rot_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Train Loss2 (Rot)')
    #     plt.subplot(413)
    #     plt.plot(outs["val_trans_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Val Loss1 (Trans)')
    #     plt.subplot(414)
    #     plt.plot(outs["val_rot_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Val Loss2 (Rot)')
    # else:
    #     plt.subplot(211)
    #     plt.plot(outs["train_trans_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Train Loss1 (Trans)')
    #     plt.subplot(212)
    #     plt.plot(outs["train_rot_loss_hist"])
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Train Loss2 (Rot)')
    # plt.tight_layout()
    # plt.show()
    
    # plot gradients
    if CHECK_GRADIENTS:
        plt.plot(outs["layer_idx"], outs["avg_grads"], marker="x")
        plt.xlabel('Layer Depth')
        plt.ylabel('Average Gradients')
        plt.tight_layout()
        plt.show()
        
    # plot accuracies
    if CHECK_ACCURACY:
        L = len(outs["rmse_train_history"])
        train_rmse = outs["rmse_train_history"] 
        val_rmse = outs["rmse_val_history"]
        
        # Loop truth and plot each output as a subplot
        plt.plot(train_rmse, '-o')
        plt.plot(val_rmse, '-o')
        plt.legend(['Train', 'Val'])
        plt.ylabel('OUTPUT RMSE')
        plt.xlabel('Epochs')
            

    
        # hitch_rmse_train = train_rmse[:,0]
        # hr_rmse_train = train_rmse[:,1]
        # hitch_rmse_val = val_rmse[:,0]
        
        # hr_rmse_val = val_rmse[:,1]
        # state3_rmse_train = train_rmse[:,2]
        # state3_rmse_val = val_rmse[:,2]
        # plt.subplot(3,1,1)
        # plt.plot(hitch_rmse_train, '-o')
        # plt.plot(hitch_rmse_val, '-o')
        # plt.legend(['train', 'val'], loc='upper right')
        # plt.ylabel('Hitch RMSE [deg]')
        # plt.xlabel('Epochs')
        # plt.subplot(3,1,2)
        # plt.plot(hr_rmse_train, '-o')
        # plt.plot(hr_rmse_val, '-o')
        # plt.ylabel('Hitch Rate RMSE [deg/s]')
        # plt.xlabel('Epochs')
        # plt.tight_layout()
        # plt.subplot(3,1,3)
        # plt.plot(state3_rmse_train, '-o')
        # plt.plot(state3_rmse_val, '-o')
        # plt.ylabel('State3 RMSE [deg/s]')
        # plt.xlabel('Epochs')
        # plt.tight_layout()
        # plt.show()