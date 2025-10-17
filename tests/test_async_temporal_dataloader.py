#%%
import os
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader

#%%
# Load data
SEQ_ROOT_PROCESSED = "D:\\TrainingData\\simulation\\processed_10Hz"
SEQ_ROOT_RAW = "D:\\TrainingData\\simulation\\processed"
SEQUENTIAL_LOOKBACK = 2

full_set = AsyncTemporalDataLoader(sequence_root_processed=SEQ_ROOT_PROCESSED,
                                   sequence_root_raw=SEQ_ROOT_RAW,
                                   sequential_lookback=SEQUENTIAL_LOOKBACK,
                                    transform_img = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((512,512)),
                                        transforms.ToTensor(),
                                    ]))

# split training set to training/val sets
NUM_VAL = 1000
NUM_TRAIN = len(full_set) - NUM_VAL
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])

# generate loaders
loader_train = DataLoader(train_set, batch_size=1, shuffle=True)
loader_val = DataLoader(val_set, batch_size=1, shuffle=True)

#%%
# Simulate using dataloader (to enter __getitem__() method)
for i_batch, sample_data in enumerate(loader_val):
    pass