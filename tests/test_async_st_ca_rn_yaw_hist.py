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
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.spacetime.async_st_ca_rn_yaw_hist import AsyncSpaceTimeCrossAttentionResNetYawHist


#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\"
SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\"            
# SEQ_ROOT_PROCESSED = "D:\\TrainingData\\simulation\\10Hz\\INT\\INT1\\"
# SEQ_ROOT_RAW = "D:\\TrainingData\\simulation\\processed\\INT\\INT1\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_yaw_hist"
WEIGHT_FILE = "async_st_ca_rn_yaw_hist_v2.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 1
NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTANT_WORKERS = True
# NUM_WORKERS = 0
# PIN_MEMORY = False
# PREFETCH_FACTOR = None
# PERSISTANT_WORKERS = False

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
NUM_DELTAS = 1
NUM_OUTPUTS = 5

#%%
def test():
    # Load dataset
    test_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        single_test=True,
        sequential_lookback=NUM_FRAMES,
        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':True},
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
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTANT_WORKERS)

    # Load model
    model = AsyncSpaceTimeCrossAttentionResNetYawHist(
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

            x[0] = x[0].to(device=device, dtype=torch.float32) # images
            x[1] = x[1].to(device=device, dtype=torch.float32) # IMU
            x[2] = x[2].to(device=device, dtype=torch.float32) # yaw history

            y = y.to(device=device, dtype=torch.float32)

            # initialize yaw estimates with truth
            if t == 0:
                pass #yaw_hist = x[2] # initialize the yaw history from external sources (from dataloader)
            elif t % 100 == 0:
                x[2] = x[2] + torch.rand_like(x[2]) * 0.01
            else: # After first pass, start using estimates as history yaw input
                x[2] = yaw_hist
        
            trans_est, rot_est, yaw_est = model(x)
            
            # update yaw hist with latest prediction and popping earlies entry
            yaw_hist = yaw_est.unsqueeze(1)
            # yaw_hist_list = yaw_hist.tolist()
            # yaw_hist_list[0].append(yaw_est.squeeze().cpu()) # appends latest estimate
            # yaw_hist_list[0].pop(0) # pops earliest entry
            # yaw_hist = torch.tensor(yaw_hist_list).to(device=device, dtype=torch.float32) # convert back to tensor
            
            est = torch.cat((trans_est, rot_est, yaw_est), dim=1)
            est_array.append(est)
            truth_array.append(y)

    est_array = torch.cat(est_array).cpu().numpy()
    truth_array = torch.cat(truth_array).cpu().numpy()
    
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
    sinyaw = est_array[:,3]
    cosyaw = est_array[:,4]
    
    dx_body_truth = truth_array[:,0]
    dy_body_truth = truth_array[:,1]
    dyaw_truth = truth_array[:,2]
    sinyaw_truth = truth_array[:,3]
    cosyaw_truth = truth_array[:,4]
    
    # df = pd.read_csv(TEST_CSV)
    df = test_set.df
    X_est_array = []
    Y_est_array = []
    X_est_array2 = []
    Y_est_array2 = []
    yaw_est_array = []
    X_est_array.insert(0,df.iloc[0]["X"])
    Y_est_array.insert(0,df.iloc[0]["Y"])
    X_est_array2.insert(0,df.iloc[0]["X"])
    Y_est_array2.insert(0,df.iloc[0]["Y"])
    yaw_est_array.insert(0,df.iloc[0]["yaw"])
    
    yaw_est_from_pred = np.arctan2(sinyaw, cosyaw)
    yaw_est_from_pred = np.unwrap((yaw_est_from_pred + 2 * np.pi) % (2 * np.pi))
    # yaw_est_from_pred.insert(0,df.iloc[0]["yaw"])
    
    for i in range(1,len(df)):
        pose_prev = (X_est_array[i-1], Y_est_array[i-1], yaw_est_array[i-1])
        X_est, Y_est =  body_to_tangent_frame_translation(pose_prev, dx_body=dx_body[i-1], dy_body=dy_body[i-1])
        yaw_est = yaw_est_array[i-1] + dyaw[i-1]
        
        X_est_array.append(X_est)
        Y_est_array.append(Y_est)
        yaw_est_array.append(yaw_est)

        # New positions from absolute yaw pred
        pose_prev2 = (X_est_array2[i-1], Y_est_array2[i-1], yaw_est_from_pred[i-1])
        X_est2, Y_est2 = body_to_tangent_frame_translation(pose_prev2, dx_body[i-1], dy_body[i-1])
        
        X_est_array2.append(X_est2)
        Y_est_array2.append(Y_est2)
        
        
    # visualize
    plt.plot(X_est_array, Y_est_array)
    plt.plot(X_est_array2, Y_est_array2)
    plt.plot(df["X"], df["Y"], '--')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Est from dyaw", "Est from pred", "Truth"])
    plt.show()

    error = compute_abs_pos_error((df["X"],df["Y"]), (X_est_array, Y_est_array))
    error2 = compute_abs_pos_error((df["X"],df["Y"]), (X_est_array2, Y_est_array2))
    plt.plot(error)
    plt.plot(error2)
    plt.legend(['From dyaw', 'From pred'])
    plt.ylabel("Position Error")
    plt.show()

    plt.subplot(211)
    plt.plot(X_est_array - df["X"])
    plt.plot(X_est_array2 - df["X"])
    plt.ylabel("Easting Error")
    plt.subplot(212)
    plt.plot(Y_est_array - df["Y"])
    plt.plot(Y_est_array2 - df["Y"])
    plt.ylabel("Northing Error")
    plt.show()

    plt.plot(np.rad2deg(df["yaw"]), '--')
    plt.plot(np.rad2deg(yaw_est_array))
    plt.plot(np.rad2deg(yaw_est_from_pred))
    plt.ylabel("Yaw prediction")
    plt.legend(["Truth", "Est from disp", "Est from pred"])
    plt.show()

    plt.subplot(511)
    plt.plot(dx_body_truth)
    plt.plot(dx_body)
    plt.legend(['Truth', 'Pred'])
    plt.xlabel('dx')
    plt.subplot(512)
    plt.plot(dy_body_truth)
    plt.plot(dy_body)
    plt.xlabel('dy')
    plt.subplot(513)
    plt.plot(dyaw_truth)
    plt.plot(dyaw)
    plt.xlabel('dyaw')
    plt.subplot(514)
    plt.plot(sinyaw_truth)
    plt.plot(sinyaw)
    plt.xlabel('sin yaw')
    plt.subplot(515)
    plt.plot(cosyaw_truth)
    plt.plot(cosyaw)
    plt.xlabel('cos yaw')
    plt.tight_layout()
    plt.show()

    plt.subplot(511)
    plt.title('Prediction Errors')
    plt.plot(dx_body_truth.squeeze() - dx_body)
    plt.subplot(512)
    plt.plot(dy_body_truth.squeeze() - dy_body)
    plt.subplot(513)
    plt.plot(np.rad2deg(dyaw_truth.squeeze() - dyaw))
    plt.subplot(514)
    plt.plot(sinyaw_truth.squeeze() - sinyaw)
    plt.subplot(515)
    plt.plot(cosyaw_truth.squeeze() - cosyaw)
    plt.tight_layout()
    plt.show()