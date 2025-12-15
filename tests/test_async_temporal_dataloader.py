#%%
import torch
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader

#%%
# Load data
SEQ_ROOT_PROCESSED = "D:\\TestingData\\experimental\\10Hz\\original\\6_19_25\\02\\"
SEQ_ROOT_RAW = "D:\\TestingData\\experimental\\40Hz\\original\\6_19_25\\02\\" 
SEQUENTIAL_LOOKBACK = 11
IMG_SIZE = (224,448)

dataset = AsyncTemporalDataLoader(
    sequence_root_processed=SEQ_ROOT_PROCESSED,
    sequence_root_raw=SEQ_ROOT_RAW,
    single_test=True,
    sequential_lookback=SEQUENTIAL_LOOKBACK,
    inputs={'cam':True, 'can':True, 'imu':True, 'yaw_hist':False},
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

# split training set to training/val sets
NUM_VAL = 1000
NUM_TRAIN = len(dataset) - NUM_VAL
# train_set, val_set = random_split(dataset,[NUM_TRAIN, NUM_VAL])

# generate loaders
loader = DataLoader(dataset, batch_size=1, shuffle=False)
# loader_val = DataLoader(val_set, batch_size=1, shuffle=False)

#%%
# Simulate using dataloader (to enter __getitem__() method)
for i_batch, sample_data in enumerate(loader):
    pass