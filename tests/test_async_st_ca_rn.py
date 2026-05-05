#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.spacetime.async_st_ca_rn import AsyncSpaceTimeCrossAttentionResNet

#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TestingData\\simulation\\10Hz\\FF\\FF1\\"
SEQ_ROOT_RAW = "D:\\TestingData\\simulation\\processed\\FF\\FF1\\" 

SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\04\\"
SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\04\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_acc_yaw\\"
WEIGHT_FILE = "async_st_ca_rn_acc_yaw_v4.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 1
VAL_RATIO = 0.2
NUM_WORKERS = 4

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
NUM_OUTPUTS = 3

#%%
# test
def test():
    # Load dataset
    test_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        sequential_lookback=SEQ_LOOKBACK,
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
        # transform_img=v2.Compose([
        #     v2.ToPILImage(),
        #     v2.Resize(IMG_SIZE),
        #     v2.ToImage(),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     ),
        # ]),
    )
    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # Load model
    model = AsyncSpaceTimeCrossAttentionResNet(
        resnet_model=torchvision.models.resnet34(weights=None),
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
    # Load weights
    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)
    
    # freeze batch norm layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            
    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(tqdm(test_loader)):

            x[0] = x[0].to(device=device, dtype=torch.float32)
            # x[1][:,:,-1] = x[1][:,:,-1] + 0.001*torch.randn_like(x[1][:,:,-1]) + 5e-4
            x[1] = x[1].to(device=device, dtype=torch.float32)

            y = y.to(device=device, dtype=torch.float32)

            trans_est, rot_est = model(x)
            
            est = torch.cat((trans_est, rot_est), dim=1)
            est_array.append(est)
            truth_array.append(y)
            
    est_array = torch.cat(est_array).squeeze().cpu().numpy()
    truth_array = torch.cat(truth_array).squeeze().cpu().numpy()
    
    print("Evaluation Complete")
    
    return est_array, truth_array, test_set

# Utility functions
def body_to_tangent_frame_translation(pose1, dx_body, dy_body):
    """
    Convert from body frame translation to tangent plane displacement
    
    Args:
        pose1: (X1, Y1, yaw1) - starting pose
        dx_body, dy_body: translation in body frame
        
    Returns:
        (dx_world, dy_world) - translation in world/tangent frame
    """
    X1, Y1, yaw1 = pose1
    
    # Rotate from body frame to world frame
    cos_yaw = np.cos(yaw1)  # Note: positive yaw1
    sin_yaw = np.sin(yaw1)  # Note: positive yaw1
    
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    
    X2 = X1 + dx_world
    Y2 = Y1 + dy_world
    
    return X2, Y2

def compute_abs_pos_error(coords1, coords2):
    X1,Y1 = coords1
    X2,Y2 = coords2
    
    error = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
    return error

#%%
# run main and plot
if __name__ == "__main__":
    
    est_array, truth_array, test_set = test()
    
    dx_body = est_array[:,0]
    dy_body = est_array[:,1]
    dyaw = est_array[:,2]
    
    dx_body_truth = truth_array[:,0]
    dy_body_truth = truth_array[:,1]
    dyaw_truth = truth_array[:,2]
    
    # df = pd.read_csv(TEST_CSV)
    df = test_set.df
    X_est_array = []
    Y_est_array = []
    yaw_est_array = []
    X_est_array.insert(0,df.iloc[0]["X"])
    Y_est_array.insert(0,df.iloc[0]["Y"])
    yaw_est_array.insert(0,df.iloc[0]["yaw"])
    
    reset_interval = 50
    for i in range(1,len(df)):
        if i == len(df) // 2: # Apply psudeo GPS correction at halfway mark
            X_est_array[i-1] = df['X'].iloc[i-1]
            Y_est_array[i-1] = df['Y'].iloc[i-1]
            yaw_est_array[i-1] = df['yaw'].iloc[i-1]
        pose_prev = (X_est_array[i-1], Y_est_array[i-1], yaw_est_array[i-1])
        X_est, Y_est =  body_to_tangent_frame_translation(pose_prev, dx_body=dx_body[i-1], dy_body=dy_body[i-1])
        yaw_est = yaw_est_array[i-1] + dyaw[i-1]
        
        X_est_array.append(X_est)
        Y_est_array.append(Y_est)
        yaw_est_array.append(yaw_est)
        
    # visualize
    plt.plot(X_est_array, Y_est_array)
    plt.plot(df["X"], df["Y"], '--')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Est", "Truth"])
    plt.show()
    
    error = compute_abs_pos_error((df["X"],df["Y"]), (X_est_array, Y_est_array))
    plt.plot(error)
    plt.ylabel("Position Error")
    plt.show()
    
    plt.subplot(211)
    plt.plot(X_est_array - df["X"])
    plt.ylabel("Easting Error")
    plt.subplot(212)
    plt.plot(Y_est_array - df["Y"])
    plt.ylabel("Northing Error")
    plt.show()
        
    plt.subplot(211)
    plt.plot(yaw_est_array)
    plt.plot(df["yaw"], '--')
    plt.ylabel("Yaw prediction")
    plt.legend(["Est", "Truth"])
    plt.subplot(212)
    plt.plot(yaw_est_array - df["yaw"])
    plt.show()
    
    plt.subplot(311)
    plt.title('Odom Predictions')
    plt.plot(dx_body_truth)
    plt.plot(dx_body)
    plt.ylabel('dx (m)')
    plt.legend(['Truth', 'Pred'])
    plt.subplot(312)
    plt.plot(dy_body_truth)
    plt.plot(dy_body)
    plt.ylabel('dy (m)')
    plt.subplot(313)
    plt.plot(np.rad2deg(dyaw_truth))
    plt.plot(np.rad2deg(dyaw))
    plt.ylabel('dyaw (deg)')
    plt.tight_layout()
    plt.show()
    
    plt.subplot(311)
    plt.title('Odom Error')
    plt.plot(dx_body_truth.squeeze() - dx_body)
    plt.ylabel('dx (m)')
    plt.subplot(312)
    plt.plot(dy_body_truth.squeeze() - dy_body)
    plt.ylabel('dy (m)')
    plt.subplot(313)
    plt.plot(np.rad2deg(dyaw_truth.squeeze() - dyaw))
    plt.ylabel('dyaw (deg)')
    plt.tight_layout()
    plt.show()
    

