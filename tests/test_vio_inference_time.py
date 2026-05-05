#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention

#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TrainingData\\experimental\\10Hz\\original\\6_19_25\\05\\"
SEQ_ROOT_RAW = "D:\\TrainingData\\experimental\\40Hz\\original\\6_19_25\\05\\"            
# SEQ_ROOT_PROCESSED = "D:\\TestingData\\simulation\\10Hz\\FF\\FF1\\"
# SEQ_ROOT_RAW = "D:\\TestingData\\simulation\\processed\\FF\\FF1\\" 

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_space_time_official\\"
WEIGHT_FILE = "exp_v1.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 1
NUM_WORKERS = 0
# PREPROCESS_DATA = { # SIM TRAINING DATA STATISTICS (IMU0)
#     "mean_steer_ang": 0.00010474232904788316, 
#     "mean_vx": 18.287964405986905,
#     "mean_imu_accel_x": -0.08054565556000937, 
#     "mean_imu_accel_y": 0.059256087349158076, 
#     "mean_imu_accel_z": -9.820629497778299, 
#     "mean_imu_gyro_x": -0.0004527679883824836, 
#     "mean_imu_gyro_y": 1.7486348199251608e-06,
#     "mean_imu_gyro_z": -0.0007029802208594473,
#     "std_steer_ang": 0.11945972354652491, 
#     "std_vx": 9.763229616274344,
#     "std_imu_accel_x": 0.38069991167038575, 
#     "std_imu_accel_y": 2.0534440012091593, 
#     "std_imu_accel_z": 0.263113227s61089984, 
#     "std_imu_gyro_x": 0.007918682043185972, 
#     "std_imu_gyro_y": 0.002558814472160346, 
#     "std_imu_gyro_z": 0.16526482988751023
# }
PREPROCESS_DATA=None
# === MODEL PARAMETERS ===
NUM_FRAMES = 2
NUM_IMU_SAMPLES = 5
EMBED_DIM = 384
NUM_HEADS = 8
DEPTH = 8
PATCH_SIZE = 16
IN_CHANNELS = 3
IMU_CHANNELS = 8
DROPOUT = 0.
NUM_OUTPUTS = 3

#%%
def test():
    # Load dataset
    test_set = AsyncTemporalDataLoader(sequence_root_processed=SEQ_ROOT_PROCESSED,
                                        sequence_root_raw=SEQ_ROOT_RAW,
                                        single_test=True,
                                        sequential_lookback=NUM_FRAMES,
                                        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
                                        # reduce={'target_column':'steer_ang', 'target_size':100},
                                        transform_img=v2.Compose([
                                            v2.ToPILImage(),
                                            v2.Resize(IMG_SIZE),
                                            v2.ToTensor(),
                                        ]),
                                        preprocess_data=PREPROCESS_DATA,
                                        # augment_imu=True,
                                    )

    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = AsyncSpaceTimeCrossAttention(IMG_SIZE,
                                         PATCH_SIZE,
                                         IN_CHANNELS,
                                         EMBED_DIM,
                                         NUM_FRAMES,
                                         NUM_IMU_SAMPLES,
                                         IMU_CHANNELS,
                                         NUM_HEADS,
                                         DEPTH,
                                         DROPOUT,
                                         )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device is Use: %s' % device)
    model = model.to(device)
    # Load weights
    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)
    
    eval_len = 1000
    # evalulate single model
    print("Evaluating Model ...")
    inference_times = []
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(tqdm(test_loader)):
            
            if t > eval_len:
                break
            x[0] = x[0].to(device=device, dtype=torch.float32)
            x[1] = x[1].to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            start_time = time.time()
            trans_est, rot_est = model(x)
            torch.cuda.synchronize()
            inf_time = time.time() - start_time
            inference_times.append(inf_time)

    print("Evaluation Complete")

    # take average of inference time
    average_inference_time = np.mean(inference_times)
    return average_inference_time

#%%
# run main and plot
if __name__ == "__main__":
    
    average_inference_time = test()
    
    print(f'Average Inference Time: {average_inference_time}')
    print(f'Resultant FPS: {1 / average_inference_time}')
# %%
