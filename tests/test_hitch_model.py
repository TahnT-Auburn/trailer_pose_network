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
from torch.utils.data import DataLoader

from trailer_pose_network.dataloaders.trailer_hitch_dataloader import HitchDataloader
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel


#%%
# Set Global variables

# === FILE LOADING ===
TEST_CSV_ROOT = "D:\\TestingData\\simulation\\10Hz\\FF\\FF2_1\\"

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\trailer_hitch"
WEIGHT_FILE = "sim_v1.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
IMG_SIZE = (224,224)
BATCH_SIZE = 6
NUM_WORKERS = 4

# === MODEL PARAMETERS ===
ENCODER = torchvision.models.mobilenet_v2(weights=None)
EMBED_DIM = 784
DROPOUT = 0.

#%%
def test():
    # Load dataset
    test_set = HitchDataloader(
        csv_root=TEST_CSV_ROOT,
        # reduce={'target_column':'steer_ang', 'target_size':10},
        transforms=v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMG_SIZE),
            v2.ToTensor(),
        ]),
    )
    
    # Generate loaders
    loader_test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Load model
    model = HitchModel(
        encoder=ENCODER,
        embed_dim=EMBED_DIM,
        dropout=DROPOUT,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device is Use: %s' % device)
    model = model.to(device)
    state_dict = torch.load(WEIGHT_PATH)
    model.load_state_dict(state_dict)

    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(tqdm(loader_test)):
            
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            hitch_est = model(x)
            
            est_array.append(hitch_est.squeeze())
            truth_array.append(y)
            
    est_array = torch.cat(est_array).cpu().numpy()
    truth_array = torch.cat(truth_array).cpu().numpy()
    
    print("Evaluation Complete")
    
    return est_array, truth_array
#%%
# Call training and visualize training
if __name__ == "__main__":
    
    est_array, truth_array = test()
    
    plt.figure()
    plt.subplot(211)
    plt.plot(np.rad2deg(truth_array), '--')
    plt.plot(np.rad2deg(est_array))
    plt.legend(['Truth', 'Pred'])
    plt.ylabel('Hitch (deg)')
    plt.subplot(212)
    plt.plot(np.rad2deg(truth_array - est_array))
    plt.ylabel('Hitch Error (deg)')