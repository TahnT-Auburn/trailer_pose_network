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
from trailer_pose_network.models.spacetime.async_st_ca_rn_trailer import AsyncSpaceTimeCrossAttentionResNet

from trailer_pose_network.trainers.trainer_acc_yaw_deltas_and_trailer import Trainer

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

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\async_st_ca_rn_acc_yaw_trailer\\imu_0\\"
WEIGHT_FILE = "async_st_ca_rn_acc_yaw_trailer_imu0_v2.pth"
WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)
SAVE_WEIGHTS = WEIGHT_SAVE_PATH

OUT_LOG_PATH = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\logs\\space_time\\async_st_ca_rn_acc_yaw_trailer\\imu_0\\sim_results2\\train_log.csv"

PRETRAINED_WEIGHTS = None
PRETRAINED = False

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 4
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTANT_WORKERS = True
# NUM_WORKERS = 0
# PIN_MEMORY = False
# PREFETCH_FACTOR = None
# PERSISTANT_WORKERS = False
# PREPROCESS_DATA = { # SIM TRAINING DATA STATISTICS (IMU0)
#     "mean_steer_ang": 0.00010474232904788316, 
#     # "mean_vx": 18.287964405986905,
#     "mean_imu_accel_x": -0.08054565556000937, 
#     "mean_imu_accel_y": 0.059256087349158076, 
#     "mean_imu_accel_z": -9.820629497778299, 
#     "mean_imu_gyro_x": -0.0004527679883824836, 
#     "mean_imu_gyro_y": 1.7486348199251608e-06,
#     "mean_imu_gyro_z": -0.0007029802208594473,
#     "std_steer_ang": 0.11945972354652491, 
#     # "std_vx": 9.763229616274344,
#     "std_imu_accel_x": 0.38069991167038575, 
#     "std_imu_accel_y": 2.0534440012091593, 
#     "std_imu_accel_z": 0.26311322761089984, 
#     "std_imu_gyro_x": 0.007918682043185972, 
#     "std_imu_gyro_y": 0.002558814472160346, 
#     "std_imu_gyro_z": 0.16526482988751023
# }
PREPROCESS_DATA = None

# === MODEL PARAMETERS ===
NUM_DELTAS = 1
NUM_FRAMES = 2
NUM_IMU_SAMPLES = 5
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
IMU_CHANNELS = 8
DROPOUT = 0.1
MODALITY_DROPOUT = {
    "imu_dropout_rate": 0.0,
    "cam_dropout_rate": 0.0
}
NUM_OUTPUTS = 5

# === TRAINING PARAMETERS ===
NUM_EPOCHS = 50
LR = 3e-5
LOSS_SCALE = [1, 1, 1, 5]
LOSS_FUNC = [nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss()]
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.005
WARMUP_PERIOD = 2 # The number of epochs to warmup
RUN_VAL = True
OVERFIT_DETECTOR = True
EARLY_STOPPING = True
CHECK_ACCURACY = True
CHECK_GRADIENTS = False

#%%
def train():
    # Load dataset
    train_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
        # reduce={'target_column':'steer_ang', 'target_size':3000},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
        ]),
        get_data_stats=True,
        preprocess_data=PREPROCESS_DATA,
    )
    # load val set
    val_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED_VAL,
        sequence_root_raw=SEQ_ROOT_RAW_VAL,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
        reduce={'target_column':'steer_ang', 'target_size':1500},
        transform_img=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
        ]),
        preprocess_data=PREPROCESS_DATA,
    )


    # Generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTANT_WORKERS)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTANT_WORKERS)
    
    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device is Use: %s' % device)
    # resent models
    # resnet_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    # resnet_hold_hitch = torchvision.models.resnet34(weights='IMAGENET1K_V1')
    # resnet_model = resnet_model.to(device)
    # resnet_hold_hitch = resnet_hold_hitch.to(device)
    
    # full model
    model = AsyncSpaceTimeCrossAttentionResNet(
        resnet_model=torchvision.models.resnet34(weights='IMAGENET1K_V1'),
        resnet_model_hitch=torchvision.models.resnet34(weights='IMAGENET1K_V1'),
        num_deltas=NUM_DELTAS,
        img_size=IMG_SIZE,
        seqential_lookback=SEQ_LOOKBACK,
        embed_dim=EMBED_DIM,
        num_frames=NUM_FRAMES,
        num_imu_samples=NUM_IMU_SAMPLES,
        imu_channels=IMU_CHANNELS,
        num_heads=NUM_HEADS,
        depth=DEPTH,
        dropout=DROPOUT,
        modality_dropout=MODALITY_DROPOUT,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Load pretrained weights if prompted
    if PRETRAINED:
        state_dict = torch.load(PRETRAINED_WEIGHTS)
        model.load_state_dict(state_dict)
        
    # Set up training
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*2, T_mult=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=(NUM_EPOCHS)*len(loader_train), eta_min=0.0)
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
        early_stopping=EARLY_STOPPING,
        loss_scale=LOSS_SCALE,
        loss_save_interval=None,
        device=device,
        check_accuracy=CHECK_ACCURACY,
        check_gradients=CHECK_GRADIENTS,
        verbose=len(loader_train),
        save_weights=SAVE_WEIGHTS,
        checkpoint_interval=None,
        save_outs=OUT_LOG_PATH,
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


    plt.figure()
    plt.suptitle('Taining Losses vs. Iteration')
    plt.subplot(411)
    plt.plot(outs["train_trans_loss_hist"])
    plt.xlabel('Iterations')
    plt.ylabel('Trans loss')
    plt.subplot(412)
    plt.plot(outs["train_rot_loss_hist"])
    plt.xlabel('Iterations')
    plt.ylabel('Rot loss')
    plt.subplot(413)
    plt.plot(outs["train_hitch_loss_hist"])
    plt.xlabel('Iterations')
    plt.ylabel('Hitch loss')
    plt.subplot(414)
    plt.plot(outs["train_acc_yaw_loss_hist"])
    plt.xlabel('Iterations')
    plt.ylabel('Acc yaw loss')
    if RUN_VAL:
        plt.figure()
        plt.subplot(411)
        plt.plot(outs["val_trans_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Trans loss')
        plt.subplot(412)
        plt.plot(outs["val_rot_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Rot loss')
        plt.subplot(413)
        plt.plot(outs["val_hitch_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Hitch loss')
        plt.subplot(414)
        plt.plot(outs["val_acc_yaw_loss_hist"])
        plt.xlabel('Iterations')
        plt.ylabel('Acc yaw loss')
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
            
            
            
            

# PREPROCESS_DATA = {     # SIM TRAINING DATA STATISTICS (IMU1)
#     "mean_steer_ang": 0.00010474232904788316, 
#     "mean_vx": 18.287964405986905, 
#     "mean_imu_accel_x": -0.03210047741594339, 
#     "mean_imu_accel_y": 0.02469601869018359, 
#     "mean_imu_accel_z": -9.815943088412155, 
#     "mean_imu_gyro_x": -0.001249379855227762, 
#     "mean_imu_gyro_y": 0.0008506330686480256, 

#     "mean_imu_gyro_z": -0.00031432984528056176,
#     "std_steer_ang": 0.11945972354652493, 
#     "std_vx": 9.763229616274344, 
#     "std_imu_accel_x": 0.258235143710354, 
#     "std_imu_accel_y": 2.0279488102074175, 
#     "std_imu_accel_z": 0.12271902157743174, 
#     "std_imu_gyro_x": 0.009022401167484217, 
#     "std_imu_gyro_y": 0.0037328788472177263, 
#     "std_imu_gyro_z": 0.165953626021579
# }
# PREPROCESS_DATA = { # CURRENTLY THE TEST DATA STATISICS (IMU1) (FOR DATA SWAP)
#     "mean_steer_ang": -0.0018256783213060977,
#     "mean_vx": 11.617652042642014,
#     "mean_imu_accel_x": -0.07951864128677506,
#     "mean_imu_accel_y": -0.08383250730508374,
#     "mean_imu_accel_z": -9.856500136352544,
#     "mean_imu_gyro_x": -0.003448715528186841,
#     "mean_imu_gyro_y": -0.0008867832780533307,
#     "mean_imu_gyro_z": -0.002825871540460926,
#     "std_steer_ang": 0.12929468914605133, 
#     "std_vx": 3.5636214327853173, 
#     "std_imu_accel_x": 0.2574701751675872, 
#     "std_imu_accel_y": 2.342593752066911, 
#     "std_imu_accel_z": 0.13390621690756863, 
#     "std_imu_gyro_x": 0.01692068461458399, 
#     "std_imu_gyro_y": 0.006417072237157534, 
#     "std_imu_gyro_z": 0.20520836409808133
# }