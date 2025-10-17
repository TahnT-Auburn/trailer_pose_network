"""
Space-Time Vision Transformer Training Script
"""
#%%
import numpy as np
import pandas
import os
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.functional as F
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

#%%
def main():
    # Generate dataloaders
    IN_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\data\\simulation"
    IN_FILE = "full_training.csv"
    TRAIN_CSV = os.path.join(IN_PARENT, IN_FILE)
    SEQ_PARENT = "D:\\TrainingData\\simulation\\processed_10Hz"

    WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\space_time"
    WEIGHT_FILE = "space_time_cross_attn_v5.pth"
    WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

    PRETRAINED_WEIGHTS = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\simulation\\spacetime\\space_time_cross_attn_v4.pth"
    PRETRAINED = False
    
    NUM_FRAMES = 2
    IMG_SIZE = (224,224)
    BATCH_SIZE = 6
    NUM_WORKERS = 4
    
    full_set = TractorTrailerData(csv_file=TRAIN_CSV,
                                    inputs={"cam":True, "can": True, "imu": True},
                                    reduce={"target_column":"yaw", "target_size":1000},
                                    transform_img=transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize(IMG_SIZE),
                                        transforms.ToTensor(),
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                        ]),
                                    transform_data=True,
                                    sequential=NUM_FRAMES,
                                    sequence_root=SEQ_PARENT,
                                    )

    # split training set to training/val sets
    NUM_VAL = int(np.round(0.2*len(full_set)))
    NUM_TRAIN = len(full_set) - NUM_VAL
    train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
    # tiny_set,val_set, remainder = random_split(full_set,[0.1,0.1,0.8])

    # generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # loader_tiny = DataLoader(tiny_set, batch_size=BATCH_SIZE, shuffle=True)

    NUM_OUTPUTS = 3
    EMBED_DIM = 384
    NUM_HEADS = 8
    NUM_LAYERS = 12
    PROJ_DROP = 0.
    ATTN_DROP = 0
    IMG_PATCH_SIZE = 16
    IMG_CHANNELS = 3
    INERT_PATCH_SIZE = 1
    INERT_CHANNELS = 1
    INERT_SIZE = (1,8)

    # Load model
    # model = SpaceTimeEncoder(embed_dim=EMBED_DIM,
    #                         num_heads=NUM_HEADS,
    #                         depth=NUM_LAYERS,
    #                         attn_drop=ATTN_DROP,
    #                         proj_drop=PROJ_DROP,
    #                         num_frames=NUM_FRAMES,
    #                         num_outputs=NUM_OUTPUTS,
    #                         inp_size=INERT_SIZE,
    #                         patch_size=INERT_PATCH_SIZE,
    #                         in_channels=INERT_CHANNELS)

    # model = SpaceTimeEncoder(embed_dim=EMBED_DIM,
    #                         num_heads=NUM_HEADS,
    #                         depth=NUM_LAYERS,
    #                         attn_drop=ATTN_DROP,
    #                         proj_drop=PROJ_DROP,
    #                         num_frames=NUM_FRAMES,
    #                         num_outputs=NUM_OUTPUTS,
    #                         inp_size=IMG_SIZE,
    #                         patch_size=IMG_PATCH_SIZE,
    #                         in_channels=IMG_CHANNELS)

    # model = SpaceTimeEncoderEarlyFusion(num_frames=NUM_FRAMES,
    #                                     embed_dim=EMBED_DIM,
    #                                     img_size=IMG_SIZE, img_patch_size=IMG_PATCH_SIZE, img_channels=IMG_CHANNELS,
    #                                     inp_size=INERT_SIZE, inp_patch_size=INERT_PATCH_SIZE, inp_channels=INERT_CHANNELS,
    #                                     num_heads=NUM_HEADS, depth=NUM_LAYERS,
    #                                     attn_drop=ATTN_DROP, proj_drop=PROJ_DROP)

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
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device in Use: %s" % device)
    model = model.to(device)

    # Implement transfer learning with pretrained weights if prompted
    if PRETRAINED:
        state_dict = torch.load(PRETRAINED_WEIGHTS)
        model.load_state_dict(state_dict)
        
    # Setup training
    NUM_EPOCHS = 10
    LR = 1e-4
    # LOSS_SCALE = [1e0, 4e1]
    # LOSS_FUNC = [nn.L1Loss(), nn.L1Loss()]
    LOSS_SCALE = 1e1
    LOSS_FUNC = nn.L1Loss()
    num_iters = len(loader_train) * NUM_EPOCHS
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*2, T_mult=2)
    warmup_period = len(loader_train)//2 
    warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)

    print("Training ...")
    print()

    network_trainer = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler,
                            loader_train=loader_train,
                            loader_val=loader_val,
                            loss_scale=LOSS_SCALE,
                            device=device,
                            check_accuracy=True,
                            verbose=len(loader_train),
                            save_weights=None)

    start_time = time.time()
    outs = network_trainer.train(loss_func=LOSS_FUNC, epochs=NUM_EPOCHS)
    print("Training Time: %s" % (time.time() - start_time))
    print()

    return outs

#%%
# Call main and visualize training
if __name__ == "__main__":
    
    outs = main()
    
    # plot loss
    plt.subplot(3,1,1)
    plt.plot(outs["lr_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    plt.subplot(3,1,2)
    plt.plot(outs["raw_loss_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.subplot(3,1,3)
    plt.plot(outs["loss_history"], '-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

    plt.subplot(211)
    plt.plot(outs["raw_loss1_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Loss1 (Translations)')
    plt.subplot(212)
    plt.plot(outs["raw_loss2_history"])
    plt.xlabel('Iterations')
    plt.ylabel('Loss2 (Rotation)')
    plt.tight_layout()
    plt.show()
    
    # plot accuracies
    L = len(outs["rmse_train_history"])
    train_rmse = np.concatenate(outs["rmse_train_history"]).reshape(L,3)
    val_rmse = np.concatenate(outs["rmse_val_history"]).reshape(L,3)
    hitch_rmse_train = train_rmse[:,0]
    hitch_rmse_val = val_rmse[:,0]
    hr_rmse_train = train_rmse[:,1]
    hr_rmse_val = val_rmse[:,1]
    state3_rmse_train = train_rmse[:,2]
    state3_rmse_val = val_rmse[:,2]
    plt.subplot(3,1,1)
    plt.plot(hitch_rmse_train, '-o')
    plt.plot(hitch_rmse_val, '-o')
    plt.legend(['train', 'val'], loc='upper right')
    plt.ylabel('Hitch RMSE [deg]')
    plt.xlabel('Epochs')
    plt.subplot(3,1,2)
    plt.plot(hr_rmse_train, '-o')
    plt.plot(hr_rmse_val, '-o')
    plt.ylabel('Hitch Rate RMSE [deg/s]')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.subplot(3,1,3)
    plt.plot(state3_rmse_train, '-o')
    plt.plot(state3_rmse_val, '-o')
    plt.ylabel('State3 RMSE [deg/s]')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()

# %%
