#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from time import sleep
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as Fs
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.data_setup import TractorTrailerData
from trailer_pose_network.models.spacetime.space_time_encoder import SpaceTimeEncoder
from trailer_pose_network.models.spacetime.space_time_early_fusion import SpaceTimeEncoderEarlyFusion
from trailer_pose_network.models.spacetime.space_time_cross_attention import SpaceTimeCrossAttention
from trailer_pose_network.trainer import Trainer
from sklearn.model_selection import train_test_split

TEST_CSV = "D:\\TrainingData\\simulation\\processed_10Hz\\HWY\\HWY2_1\\HWY2_1.csv"
SEQ_PARENT = "D:\\TrainingData\\simulation\\processed_10Hz\\HWY\\HWY2_1"
WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\space_time\\space_time_cross_attn_v4.pth"
    
def main():

    # data loader
    NUM_FRAMES = 2
    IMG_SIZE = (224,224)
    BATCH_SIZE = 6
    NUM_WORKERS = 4

    test_set = TractorTrailerData(csv_file=TEST_CSV,
                                inputs={"cam":True, "can": True, "imu": True},
                                single_test=True,
                                sequential=NUM_FRAMES,
                                sequence_root=SEQ_PARENT,
                                transform_img=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(IMG_SIZE),
                                        transforms.ToTensor(),
                                    ]),
                                transform_data=True,)


    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    L = len(test_set)

    # model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    NUM_OUTPUTS = 3
    EMBED_DIM = 768
    NUM_HEADS = 8
    NUM_LAYERS = 12
    PROJ_DROP = 0.
    ATTN_DROP = 0
    IMG_PATCH_SIZE = 16
    IMG_CHANNELS = 3
    INERT_PATCH_SIZE = 1
    INERT_CHANNELS = 1
    INERT_SIZE = (1,8)
        
    # model = SpaceTimeEncoder(num_frames=NUM_FRAMES,
    #                             inp_size=(1,4),
    #                             patch_size=1,
    #                             in_channels=1,
    #                             embed_dim=384)

    model = SpaceTimeCrossAttention(num_outputs=NUM_OUTPUTS,
                                    img_size=IMG_SIZE,
                                    patch_size=IMG_PATCH_SIZE,
                                    in_channels=IMG_CHANNELS,
                                    imu_channels=INERT_SIZE[1],
                                    embed_dim=EMBED_DIM,
                                    num_frames=NUM_FRAMES,
                                    num_heads=NUM_HEADS,
                                    depth=NUM_LAYERS,
                                    dropout=0.
                                    )
        
    # model = SpaceTimeEncoderEarlyFusion(num_frames=num_frames,
    #                                     embed_dim=384,
    #                                     img_size=(img_size,img_size), img_patch_size=16, img_channels=3,
    #                                     inp_size=(1,4), inp_patch_size=1, inp_channels=1,
    #                                     num_heads=8, depth=12)
    model = model.to(device)
    state_dict = torch.load(WEIGHTS)
    model.load_state_dict(state_dict)

    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(tqdm(test_loader)):
            # sleep(0.01)
            # start_time = time.time()
            if isinstance(x,list):
                if len(x) == 2:
                    x[0] = x[0].to(device=device, dtype=torch.float32)
                    x[1] = x[1].to(device=device, dtype=torch.float32)
                else:
                    x = x[0] # grab first TODO: Modify this to be more interactive. Make num_inputs a parameter
                    x = x.to(device=device, dtype=torch.float32)
            else:
                x = x.to(device=device, dtype=torch.float32)

            y = y.to(device=device, dtype=torch.float32)

            est = model(x)

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
    
    est_array, truth_array, test_set = main()
    
    dx_body = est_array[:,0]
    dy_body = est_array[:,1]
    dyaw = est_array[:,2]
    
    # df = pd.read_csv(TEST_CSV)
    df = test_set.df
    X_est_array = []
    Y_est_array = []
    yaw_est_array = []
    X_est_array.insert(0,df.iloc[0]["X"])
    Y_est_array.insert(0,df.iloc[0]["Y"])
    yaw_est_array.insert(0,df.iloc[0]["yaw"])
    
    for i in range(1,len(df)):
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
    plt.show()
    plt.subplot(212)
    plt.plot(Y_est_array - df["Y"])
    plt.ylabel("Northing Error")
    plt.show()
        
    plt.plot(yaw_est_array)
    plt.plot(df["yaw"], '--')
    plt.ylabel("Yaw prediction")
    plt.legend(["Est", "Truth"])
    plt.show()
    

# %%
