#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('ipympl')
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as T
from torch.utils.data import DataLoader

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader
from trailer_pose_network.models.visual_odometry.raft_based_visual_odom_model import RaftBasedVisualOdom

#%%
# Set Global variables

# === FILE LOADING ===
SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\"
SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\"            
# SEQ_ROOT_PROCESSED = "D:\\TrainingData\\simulation\\10Hz\\INT\\INT1\\"
# SEQ_ROOT_RAW = "D:\\TrainingData\\simulation\\processed\\INT\\INT1\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\experimental\\visual_odometry"
WEIGHT_FILE = "raft_based_visual_odom_v1.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 6
NUM_WORKERS = 4

# === MODEL PARAMETERS ===
EMBED_DIM = 384

def test():
    test_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        single_test=True,
        sequential_lookback=NUM_FRAMES,
        inputs={'cam':True, 'can':False, 'imu':False, 'yaw_hist':False},
        transform_img=T.Compose([
                T.ToPILImage(),
                T.Resize(IMG_SIZE),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(mean=0.5, std=0.5),
            ]),
        )
    # Generate loaders
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # load model and pretrained weights
    model = RaftBasedVisualOdom(
        embed_dim=EMBED_DIM
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # freeze batch norm layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.weight.requires_grad = False
            module.bias.requires_grad = False
            
    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)
    
    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        for t, (x,y) in enumerate(tqdm(test_loader)):

            x[0] = x[0].to(device=device, dtype=torch.float32) # images

            y = y.to(device=device, dtype=torch.float32)

            rot_est = model(x)

            # reconstruct delta yaw from sin/cos components
            delta_yaw_est = torch.atan2(rot_est[:,0], rot_est[:,1])
            delta_yaw_truth = torch.atan2(y[:,0], y[:,1])
            est_array.append(delta_yaw_est)
            truth_array.append(delta_yaw_truth)
            
    est_array = torch.cat(est_array).cpu().numpy()
    truth_array = torch.cat(truth_array).cpu().numpy()
    
    print("Evaluation Complete")

    return est_array, truth_array, test_set

if __name__ == "__main__":
    
    est_array, truth_array, test_set = test()
    
    plt.figure()
    plt.plot(np.rad2deg(truth_array))
    plt.plot(np.rad2deg(est_array))
    plt.ylabel('Deg')
    plt.title("Delta Yaw prediction")
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(np.rad2deg(truth_array)- np.rad2deg(est_array))
    plt.plot()
    plt.ylabel('Deg')
    plt.title("Delta Yaw prediction Error")
    plt.tight_layout()
    plt.show()