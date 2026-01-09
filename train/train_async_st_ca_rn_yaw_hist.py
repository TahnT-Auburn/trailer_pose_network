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
from trailer_pose_network.models.spacetime.async_st_ca_rn_yaw_hist import AsyncSpaceTimeCrossAttentionResNetYawHist

from trailer_pose_network.trainers.trainer_closed_loop_deltas_plus_yaw_hist import Trainer

#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TrainingData\\experimental\\10Hz\\original\\"
SEQ_ROOT_RAW = "D:\\TrainingData\\experimental\\40Hz\\original\\"

SEQ_ROOT_PROCESSED_VAL = "D:\\TestingData\\experimental\\10Hz\\original\\"
SEQ_ROOT_RAW_VAL = "D:\\TestingData\\experimental\\40Hz\\original\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_yaw_hist"
WEIGHT_FILE = "async_st_ca_rn_yaw_hist_v3.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)
SAVE_WEIGHTS = None

PRETRAINED_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_yaw_hist\\async_st_ca_rn_yaw_hist_v1.pth"
PRETRAINED = False

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 11
IMG_SIZE = (224,448)
BATCH_SIZE = 4
VAL_RATIO = 0.24
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTANT_WORKERS = True
# NUM_WORKERS = 0
# PIN_MEMORY = False
# PREFETCH_FACTOR = None
# PERSISTANT_WORKERS = False

# === MODEL PARAMETERS ===
NUM_DELTAS = 1
NUM_FRAMES = 2
NUM_IMU_SAMPLES = 5
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
PATCH_SIZE = 16

IN_CHANNELS = 3
IMU_CHANNELS = 8
DROPOUT = 0.
NUM_OUTPUTS = 6

# === TRAINING PARAMETERS ===
NUM_EPOCHS = 10
LR = 1e-4
LOSS_SCALE = [1, 1, 2, 5]
LOSS_FUNC = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()]
# LOSS_SCALE = 1e1
# LOSS_FUNC = nn.L1Loss()
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
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
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':True},
        reduce={'target_column':'yaw', 'target_size':100},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    )
    # load val set
    val_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED_VAL,
        sequence_root_raw=SEQ_ROOT_RAW_VAL,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':True},
        reduce={'target_column':'yaw', 'target_size':100},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    )

    
    # Generate loaders
    loader_train = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTANT_WORKERS
    )
    loader_val = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTANT_WORKERS
    )
    
    # Load model
    model = AsyncSpaceTimeCrossAttentionResNetYawHist(
        resnet_model=torchvision.models.resnet34(weights='IMAGENET1K_V1'),
        num_deltas=NUM_DELTAS,
        img_size=IMG_SIZE,
        seqential_lookback=SEQ_LOOKBACK,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_frames=NUM_FRAMES,
        num_imu_samples=NUM_IMU_SAMPLES,
        imu_channels=IMU_CHANNELS,
        num_heads=NUM_HEADS,
        depth=DEPTH,
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*10, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=(NUM_EPOCHS - WARMUP_PERIOD)*len(loader_train), eta_min=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=NUM_EPOCHS*len(loader_train), eta_min=0.0 )
    warmup_period = len(loader_train) * WARMUP_PERIOD
    warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)
    
    network_trainer = Trainer(
        model=model,
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
        save_weights=SAVE_WEIGHTS
    )
    
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
        plt.figure()
        plt.subplot(4,1,1)
        plt.plot(outs["lr_history"])
        plt.xlabel('Iterations')
        plt.ylabel('Learning Rate')
        plt.subplot(4,1,2)
        plt.plot(outs["train_total_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Training Loss')
        plt.subplot(4,1,3)
        plt.plot(outs["val_total_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Validation Loss')
        plt.subplot(4,1,4)
        plt.plot(outs["train_epoch_loss_history"], '-o')
        plt.plot(outs["val_epoch_loss_history"])
        plt.legend(["Train", "Val"])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
    else:
        plt.figure()
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

    if RUN_VAL:
        plt.figure()
        plt.subplot(411)
        plt.title('Train losses')
        plt.plot(outs["train_trans_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss1 (Trans)')
        plt.subplot(412)
        plt.plot(outs["train_rot_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss2 (Rot)')
        plt.subplot(413)
        plt.plot(outs["train_abs_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss3 (Abs yaw)')
        plt.subplot(414)
        plt.plot(outs["train_acc_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss4 (Acc yaw)')
        
        plt.figure()
        plt.subplot(411)
        plt.title('Val Losses')
        plt.plot(outs["val_trans_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Val Loss1 (Trans)')
        plt.subplot(412)
        plt.plot(outs["val_rot_lost_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Val Loss2 (Rot)')
        plt.subplot(413)
        plt.plot(outs["val_acc_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Val Loss3 (Acc yaw)')
        plt.subplot(414)
        plt.plot(outs["val_acc_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Val Loss3 (Acc yaw)')
    else:
        plt.subplot(411)
        plt.plot(outs["train_trans_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss1 (Trans)')
        plt.subplot(412)
        plt.plot(outs["train_rot_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss2 (Rot)')
        plt.subplot(413)
        plt.plot(outs["train_abs_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss3 (Abs yaw)')
        plt.subplot(414)
        plt.plot(outs["train_acc_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Train Loss3 (Acc yaw)')
    plt.tight_layout()
    plt.show()
    
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