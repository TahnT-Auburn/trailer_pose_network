#%%
import numpy as np
import os
from tqdm import tqdm

import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader

from trailer_pose_network.models.spacetime.finalized.async_space_time_cross_attention import AsyncSpaceTimeCrossAttention
from trailer_pose_network.models.spacetime.finalized.trailer_hitch_model import HitchModel
from trailer_pose_network.dataloaders.async_temporal_splits_dataloader import AsyncTemporalDataLoader

# === FILE LOADING ===
# SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\04\\"
# SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\04\\"            
SEQ_ROOT_PROCESSED = "D:\\TestingData\\simulation\\10Hz\\"
SEQ_ROOT_RAW = "D:\\TestingData\\simulation\\processed\\" 

WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\async_space_time_official\\"
WEIGHT_FILE = "sim_v0.pth"
WEIGHT_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,224)
BATCH_SIZE = 6
NUM_WORKERS = 0
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

# === LOAD HITCH MODEL ===
HITCH_WEIGHTS= "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\trailer_hitch\\sim_v1.pth"

# === MODEL PARAMETERS ===
ENCODER = torchvision.models.mobilenet_v2(weights=None)
HITCH_EMBED_DIM = 784
DROPOUT = 0.

def test():
    
    split_criterias = [
        ('TIME', 'D'), ('TIME', 'N'), ('TIME', 'M'),
        ('SETTING', 'R'), ('SETTING', 'U'), ('SETTING', 'M')
    ]
    results = {}
    
    for split_criteria in split_criterias:
        
        # Load dataset
        test_set = AsyncTemporalDataLoader(sequence_root_processed=SEQ_ROOT_PROCESSED,
                                            sequence_root_raw=SEQ_ROOT_RAW,
                                            single_test=False,
                                            split_criteria=split_criteria,
                                            sequential_lookback=NUM_FRAMES,
                                            inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
                                            reduce={'target_column':'steer_ang', 'target_size':300},
                                            transform_img=v2.Compose([
                                                v2.ToPILImage(),
                                                v2.Resize(IMG_SIZE),
                                                v2.ToTensor(),
                                            ]),
                                            preprocess_data=PREPROCESS_DATA,
                                            # augment_imu=True,
                                        )

        # Generate loaders
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

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
        ENCODER = torchvision.models.mobilenet_v2(weights=None)
        hitch_model = HitchModel(
            encoder=ENCODER,
            embed_dim=HITCH_EMBED_DIM,
            dropout=DROPOUT,
        )
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device is Use: %s' % device)
        model = model.to(device)
        hitch_model = hitch_model.to(device)
        # Load weights
        state_dict = torch.load(WEIGHT_PATH)
        model.load_state_dict(state_dict)
        hitch_state_dict = torch.load(HITCH_WEIGHTS)
        hitch_model.load_state_dict(hitch_state_dict)
        
        # evalulate single model
        est_array = []
        truth_array = []
        print("Evaluating Model ...")
        with torch.no_grad():
            model.eval()
            hitch_model.eval()
            for t, (x,y) in enumerate(tqdm(test_loader)):
                
                x[0] = x[0].to(device=device, dtype=torch.float32)
                x[1] = x[1].to(device=device, dtype=torch.float32)
                curr_images = x[0][:,-1]
                curr_images = curr_images.to(device=device, dtype=torch.float32)
                
                y = y.to(device=device, dtype=torch.float32)

                # VIO
                trans_est, rot_est = model(x)
                
                # HITCH
                hitch_est = hitch_model(curr_images)
                
                est = torch.cat((trans_est, rot_est, hitch_est), dim=1)
                est_array.append(est)
                truth_array.append(y)
                
        est_array = torch.cat(est_array).cpu().numpy()
        truth_array = torch.cat(truth_array).cpu().numpy()

        # compute RMSEs
        trans_truth = truth_array[:,0:2]
        rot_truth = truth_array[:,2:3]
        hitch_truth = truth_array[:,3:]
        
        trans_est = est_array[:,0:2]
        rot_est = est_array[:,2:3]
        hitch_est = est_array[:,3:]
        
        trans_rmse = rmse(trans_truth, trans_est)
        rot_rmse = np.rad2deg(rmse(rot_truth, rot_est))
        hitch_rmse = np.rad2deg(rmse(hitch_truth, hitch_est))
        
        # populate results
        print(f'Populating Results for {split_criteria[0], split_criteria[1]}')
        rmse_dict = {'trans_rmse': trans_rmse, 
                     'rot_rmse': rot_rmse, 
                     'hitch_rmse': hitch_rmse}
        results[split_criteria[0] + split_criteria[1]] = rmse_dict
        
    print("Evaluation Complete")

    return results

def rmse(x_true, x_pred):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return np.sqrt(np.mean((x_true - x_pred)**2))
 
if __name__ == "__main__":
    results = test()

    
    