#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
from decimal import Decimal

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split
import pytorch_warmup as warmup
from concurrent.futures import ThreadPoolExecutor

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.spacetime.async_st_ca_rn import AsyncSpaceTimeCrossAttentionResNet

from trailer_pose_network.trainer import Trainer

#%%
# Set Global variables

# === FILE LOADING ===
TEST_CSV = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\02.csv"
IMG_CSV = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\02.csv"
test_df = pd.read_csv(TEST_CSV, header="infer")
img_df = pd.read_csv(IMG_CSV, header='infer')
# SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\"
# SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\"   

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\async_st_ca_rn_acc_yaw"
WEIGHT_FILE = "async_st_ca_rn_acc_yaw_v1.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
SEQ_LOOKBACK = 6
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
NUM_OUTPUTS = 3

#%% generate truth
def gen_truth():
    def body_disp_meas_sim(N,E,yaw,sigmas=np.array([0,0,0]),biases=np.array([0,0,0])):
        """
        Generates simulated planar body frame displacement measurements which includes
        a delta x, delta y translations and delta yaw rotation in the body frame.

        Args:
            N (array-like): North positions.
            E (array-like): East positions.
            yaw (array-like): Yaw angles in radians.
            sigmas (array-like): 3 element sigmas to apply white noise to odom measurements.
            biases (array-like): 3 element biases to apply to odom measurements.
        Returns:
            dx_body (array-like): Delta x translation.
            dy_body (array-like): Delta y translation.
            dyaw (array-like): Delta yaw rotation in radians.
        """
        L = len(N)
        dx_body = []
        dy_body = []
        dyaw = []
        for i in range(1,L):
            pose_prev = (E[i-1], N[i-1], yaw[i-1])
            pose_current = (E[i], N[i], yaw[i])
            dx_body_, dy_body_, dyaw_ = tangent_to_body_frame_translation(pose_prev, pose_current)
            dx_body.append(dx_body_)
            dy_body.append(dy_body_)
            dyaw.append(dyaw_)
        # apply white noise
        L_disp = len(dx_body)
        dx_body = np.array(dx_body) + sigmas[0]*np.random.randn(L_disp) + biases[0]
        dy_body = np.array(dy_body) + sigmas[1]*np.random.randn(L_disp) + biases[1]
        dyaw = np.array(dyaw) + sigmas[2]*np.random.randn(L_disp) + biases[2]
        
        return dx_body, dy_body, dyaw


    def tangent_to_body_frame_translation(pose1, pose2):
            """
            Convert from tangent plane poses to body frame relative translation
            
            Args:
                pose1: (X1, Y1, yaw1) - starting pose
                pose2: (X2, Y2, yaw2) - ending pose
                
            Returns:
                (dx_body, dy_body) - translation in body frame of pose1
            """
            X1, Y1, yaw1 = pose1
            X2, Y2, yaw2 = pose2
            
            # World frame displacement
            dx_world = X2 - X1
            dy_world = Y2 - Y1
            
            # Rotate into body frame of pose1
            cos_yaw = np.cos(-yaw1)
            sin_yaw = np.sin(-yaw1)
            
            dx_body = cos_yaw * dx_world - sin_yaw * dy_world
            dy_body = sin_yaw * dx_world + cos_yaw * dy_world
            
            # Relative yaw change
            dyaw = yaw2 - yaw1
            
            # Normalize yaw to [-pi, pi]
            dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
            
            return dx_body, dy_body, dyaw
    # find and cap df for the final whole number (since loop setup will only provide estimates at 1Hz intervals)
    time_series = test_df['t']
    whole_number_indicies = time_series[time_series % 1 == 0].index
    last_whole_number_index = whole_number_indicies[-1]
    capped_df = test_df.iloc[:last_whole_number_index+1]
    N_etal = capped_df['Y']
    E_etal = capped_df['X']
    yaw_etal = capped_df['yaw']
    t = capped_df['t']
    mask_10hz = [Decimal(str(t_)) % Decimal('0.1') == 0 for t_ in t]
    N_etal_10hz = N_etal[mask_10hz].reset_index(drop=True)
    E_etal_10hz = E_etal[mask_10hz].reset_index(drop=True)
    yaw_etal_10hz = yaw_etal[mask_10hz].reset_index(drop=True)
    dx_body_truth, dy_body_truth, dyaw_truth = body_disp_meas_sim(
    N_etal_10hz.to_numpy(),
    E_etal_10hz.to_numpy(),
    yaw_etal_10hz.to_numpy(),
)
    truth_array = np.array([[dx_body_truth], [dy_body_truth], [dyaw_truth]]).squeeze()
    return truth_array, capped_df

#%% prepare custom dataloader
def load_data(iter, transform_img):
    """Manually prepares data similar to how a torch dataloader would.
    """
    
    # heler image loader
    def load_image(image_path):
        """Loads an image using OpenCV."""
        try:
                img = cv2.imread(image_path)
                if img is None:
                        print(f"Error: Could not read image at {image_path}")
                        return None
                return img
        except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
            
    # generate sequential block for core data
    seq_block = test_df.iloc[iter-40:iter+1] # hard code a 1 second interval given 40Hz samples
    # generate sequential block for image data
    t_to_find = [seq_block['t'].iloc[0], seq_block['t'].iloc[-1]]
    mask = img_df['t'].isin(t_to_find)
    valid_indices = img_df['t'][mask].index.tolist()
    seq_block_img = img_df.iloc[valid_indices[0]:valid_indices[-1]+1]
    
    # load images
    left_images = []
    right_images =[]
    left_paths = [seq_block_img["LRMC"].iloc[0], seq_block_img["LRMC"].iloc[5], seq_block_img["LRMC"].iloc[-1]] # NOTE: grabs first, middle, and last images ONLY!
    right_paths = [seq_block_img["RRMC"].iloc[0], seq_block_img["RRMC"].iloc[5], seq_block_img["RRMC"].iloc[-1]]
    # left_paths = seq_block["LRMC"].to_list()
    # right_paths = seq_block["RRMC"].to_list()
    with ThreadPoolExecutor(max_workers=32) as executor:
        left_images = list(executor.map(load_image,left_paths))
        right_images = list(executor.map(load_image,right_paths))

        image_pairs = list(zip(right_images, left_images))
        concat_images = list(executor.map(cv2.hconcat,image_pairs))
        # concat_images = list(executor.map(self.change_img_color,concat_images))
        if transform_img is not None:
                concat_images = list(executor.map(transform_img,concat_images))

    input_cam = torch.stack(concat_images)
    # input_cam = torch.randn(3,3,224,448)
    # emulate a batch size of 1
    input_cam = input_cam.unsqueeze(dim=0)
    
    # CAN/IMU inputs
    input_can = torch.stack([torch.tensor(seq_block["steer_ang"].to_list()),torch.tensor(seq_block["vx"].to_list())]).permute(1,0)
    input_imu = torch.stack([torch.tensor(seq_block["imu_accel_x"].to_list()),torch.tensor(seq_block["imu_accel_y"].to_list()),torch.tensor(seq_block["imu_accel_z"].to_list()),
        torch.tensor(seq_block["imu_gyro_x"].to_list()),torch.tensor(seq_block["imu_gyro_y"].to_list()),torch.tensor(seq_block["imu_gyro_z"].to_list())]).permute(1,0)
    input_inertial = torch.cat((input_can, input_imu), dim=1)
    # input_inertial = torch.randn(41,8)
    # emulate a batch size of 1
    input_inertial = input_inertial.unsqueeze(dim=0)
    
    # populate final inputs list (what goes into the model)
    inputs = [input_cam, input_inertial]
    
    return inputs


def test():
    gen_truth()
    
    # data loader param
    transform_img = v2.Compose([
        v2.ToPILImage(),
        v2.Resize(IMG_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
        # Load model
    model = AsyncSpaceTimeCrossAttentionResNet(
        resnet_model=torchvision.models.resnet34(weights=None),
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
        for k in tqdm(range(0,len(test_df))):
            if k != 0 and test_df['t'].iloc[k] % 0.5 == 0: # only query at 0.5 second intervals
                iter = k
                # call data loader function
                inputs = load_data(iter, transform_img)
                x = inputs.copy()
                x[0] = x[0].to(device=device, dtype=torch.float32)
                x[1] = x[1].to(device=device, dtype=torch.float32)

                trans_est, rot_est = model(x)
            
                est = torch.cat((trans_est.squeeze(dim=0), rot_est.squeeze(dim=0)), dim=0)
                est_array.append(est)
                # truth_array.append()
            
    est_array = torch.cat(est_array, dim=1).cpu().numpy()
    # truth_array = torch.cat(truth_array).cpu().numpy()
    
    print("Evaluation Complete")

    # generate truth
    truth_array, capped_df = gen_truth()
    
    return est_array, truth_array, capped_df

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
    
    est_array, truth_array, capped_df = test()
    
    dx_body = est_array[0,:]
    dy_body = est_array[1,:]
    dyaw = est_array[2,:]
    
    dx_body_truth = truth_array[0,:]
    dy_body_truth = truth_array[1,:]
    dyaw_truth = truth_array[2,:]
    
    # df = pd.read_csv(TEST_CSV)
    time_series = img_df['t']
    whole_number_indicies = time_series[time_series % 1 == 0].index
    last_whole_number_index = whole_number_indicies[-1]
    capped_df = img_df.iloc[:last_whole_number_index]
    df = capped_df.copy()
    X_est_array = []
    Y_est_array = []
    yaw_est_array = []
    X_est_array.insert(0,df.iloc[0]["X"])
    Y_est_array.insert(0,df.iloc[0]["Y"])
    yaw_est_array.insert(0,df.iloc[0]["yaw"])
    
    for i in range(1,len(dx_body)):
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
        
    plt.plot(yaw_est_array)
    plt.plot(df["yaw"], '--')
    plt.ylabel("Yaw prediction")
    plt.legend(["Est", "Truth"])
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
    plt.plot(dx_body_truth - dx_body)
    plt.ylabel('dx (m)')
    plt.subplot(312)
    plt.plot(dy_body_truth - dy_body)
    plt.ylabel('dy (m)')
    plt.subplot(313)
    plt.plot(np.rad2deg(dyaw_truth - dyaw))
    plt.ylabel('dyaw (deg)')
    plt.tight_layout()
    plt.show()