#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.spacetime.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention
from trailer_pose_network.models.spacetime.async_space_time_ca_yaw_hist import AsyncSpaceTimeYawHist

from trailer_pose_network.trainers.trainer_async_space_time_ca_yaw_hist import Trainer

#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\04\\"
SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\04\\"            
# SEQ_ROOT_PROCESSED = "D:\\TrainingData\\simulation\\10Hz\\INT\\INT1\\"
# SEQ_ROOT_RAW = "D:\\TrainingData\\simulation\\processed\\INT\\INT1\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_space_time_yaw_hist"
WEIGHT_FILE = "async_space_time_yaw_hist_v1.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 6
NUM_WORKERS = 4

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
NUM_OUTPUTS = 5

#%%
def test():
    # Load dataset
    test_set = AsyncTemporalDataLoader(sequence_root_processed=SEQ_ROOT_PROCESSED,
                                        sequence_root_raw=SEQ_ROOT_RAW,
                                        single_test=True,
                                        sequential_lookback=NUM_FRAMES,
                                        inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':True},
                                        # reduce={'target_column':'steer_ang', 'target_size':100},
                                        transform_img=v2.Compose([
                                            v2.ToPILImage(),
                                            v2.Resize(IMG_SIZE),
                                            v2.ToTensor(),
                                        ]),
                                    )

    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    model = AsyncSpaceTimeYawHist(IMG_SIZE,
                                         PATCH_SIZE,
                                         IN_CHANNELS,
                                         EMBED_DIM,
                                         NUM_FRAMES,
                                         NUM_IMU_SAMPLES,
                                         IMU_CHANNELS,
                                         NUM_HEADS,
                                         DEPTH,
                                         DROPOUT,
                                         NUM_OUTPUTS)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device is Use: %s' % device)
    model = model.to(device)
    # Load weights
    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)
    
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
            if t != 0: # After first pass, start using estimates as history yaw input. Assumes first 5 are free
                pass
                    
            trans_est, rot_est, yaw_est = model(x)
            
            est = torch.cat((trans_est, rot_est, yaw_est), dim=1)
            est_array.append(est)
            truth_array.append(y)
            
            # print(f"Percent Complete: {(t*BATCH_SIZE/L)*100}")
            # print(f"Single loop time: {time.time()-start_time}")
            stop=1
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
    
    yaw_est_from_pred = np.unwrap(np.arctan2(sinyaw, cosyaw)).tolist()
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
    plt.ylabel("Easting Error")
    plt.subplot(212)
    plt.plot(Y_est_array - df["Y"])
    plt.ylabel("Northing Error")
    plt.show()
    
    plt.plot(df["yaw"], '--')
    plt.plot(yaw_est_array)
    plt.plot(yaw_est_from_pred)
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
    plt.plot(dx_body_truth - dx_body)
    plt.title('Prediction Errors')
    plt.subplot(512)
    plt.plot(dy_body_truth - dy_body)
    plt.subplot(513)
    plt.plot(dyaw_truth - dyaw)
    plt.subplot(514)
    plt.plot(sinyaw_truth - sinyaw)
    plt.subplot(515)
    plt.plot(cosyaw_truth - cosyaw)
    plt.tight_layout()
    plt.show()