#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_warmup as warmup

from trailer_pose_network.trainer import Trainer
from trailer_pose_network.dataloaders.flownet_dataloader import FLownetData
from trailer_pose_network.models.vio.vanilla_vio_transformer import VanillaVIOTransformer

#%%
# Set Paths
TEST_CSV = "D:\\TestingData\\simulation\\processed\\FF\\FF1\\FF1.csv"
SEQ_PARENT = "D:\\TestingData\\simulation\\processed\\FF\\FF1"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\vio\\"
WEIGHT_FILE = "vanilla_vio_v3.pth"
WEIGHTS = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# Generate Dataloader
NUM_FRAMES = 2
IMG_SIZE = (384,512)
BATCH_SIZE = 6
    
def main():
    test_set = FLownetData(csv_file=TEST_CSV,
                        single_test=True,
                        transform=transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(IMG_SIZE),
                            transforms.ToTensor()
                        ]),
                        sequential=NUM_FRAMES,
                        sequence_root=SEQ_PARENT)


    # generate loaders
    BATCH_SIZE = 24
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    L = len(test_set)

    # Load Model

    # set model parameters
        # set model parameters
    VIS_ENCODER_PARAMS ={
        "weights": "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\flownet\\FlowNetPytorch\\weights\\flownetc_EPE1.766.pth",
        "img_size": IMG_SIZE,
        "patch_size": 1,
    }

    INERT_ENCODER_PARAMS = {
        "num_features": 8,
        "num_sequence": 2,
        "drop_out": 0.
    }

    EMBED_DIM = 384
    NUM_HEADS = 8
    NUM_LAYERS = 12
    PROJ_DROP = 0.
    ATTN_DROP = 0.

    model = VanillaVIOTransformer(embed_dim=EMBED_DIM,
                                num_heads=NUM_HEADS,
                                num_layers=NUM_LAYERS,
                                proj_drop=PROJ_DROP,
                                attn_drop=ATTN_DROP,
                                vis_encoder_params=VIS_ENCODER_PARAMS,
                                inert_encoder_params=INERT_ENCODER_PARAMS,
                                num_outputs=3)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device in Use: %s" % device)
    model = model.to(device)
    state_dict = torch.load(WEIGHTS)
    model.load_state_dict(state_dict)

    # evalulate model
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
    
    return est_array, truth_array

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
    
    est_array, truth_array = main()
    
    dx_body = est_array[:,0]
    dy_body = est_array[:,1]
    dyaw = est_array[:,2]
    
    df = pd.read_csv(TEST_CSV)
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
    
    plt.plot(yaw_est_array)
    plt.plot(df["yaw"])
    plt.ylabel("Yaw prediction")
    plt.show()
    
    error = compute_abs_pos_error((df["X"],df["Y"]), (X_est_array, Y_est_array))
    plt.plot(error)
    plt.ylabel("Position Error")
    plt.show()
    # plots
    
    # hitch_est = est_array[:,0]
    # hr_est = est_array[:,1]
    # hitch_truth = truth_array[:,0]
    # hr_truth = truth_array[:,1]

    # hitch_est = np.rad2deg(hitch_est)
    # hr_est = np.rad2deg(hr_est)
    # hitch_truth = np.rad2deg(hitch_truth)
    # hr_truth = np.rad2deg(hr_truth)
        
    # plt.subplot(211)
    # plt.plot(hitch_truth,'r')
    # plt.plot(hitch_est,'k')
    # plt.legend(["Truth", "Est"])
    # plt.xlabel('Frames')
    # plt.ylabel('Hitch [deg]')
    # plt.subplot(212)
    # plt.plot(hitch_truth - hitch_est, 'k')
    # plt.xlabel('Frames')
    # plt.ylabel('Error [deg]')
    # plt.tight_layout()
    # plt.show()

    # plt.subplot(211)
    # plt.plot(hr_truth,'r')
    # plt.plot(hr_est,'k')
    # plt.legend(["Truth", "Est"])
    # plt.xlabel('Frames')
    # plt.ylabel('Hitch Rate [deg/s]')
    # plt.subplot(212)
    # plt.plot(hr_truth - hr_est, 'k')
    # plt.xlabel('Frames')
    # plt.ylabel('Error [deg/s]')
    # plt.tight_layout()
    # plt.show()
    
    