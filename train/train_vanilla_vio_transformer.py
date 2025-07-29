'''
#####################################################
Training Script for the vanilla VIO transformer.
Model located under: models/vio/vanilla_vio_transformer.py

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

#####################################################
'''
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
def main():
    # Set Paths
    TRAIN_CSV = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\data\\simulation\\full_training.csv"
    SEQ_PARENT = "D:\\TrainingData\\simulation\\processed"

    WEIGHT_PARENT = "C:\\Users\\Tahn\\SoftDevel\\trailer_pose_network\\weights\\vio\\"
    WEIGHT_FILE = "vanilla_vio_v1.pth"
    WEIGHT_SAVE_PATH = os.path.join(WEIGHT_PARENT, WEIGHT_FILE)

    # Generate Dataloader
    NUM_FRAMES = 2
    IMG_SIZE = (384,512)
    BATCH_SIZE = 6

    full_set = FLownetData(csv_file=TRAIN_CSV,
                        reduce={"target_column":"yaw", "target_size":10000},
                        transform=transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(IMG_SIZE),
                            transforms.ToTensor()
                        ]),
                        sequential=NUM_FRAMES,
                        sequence_root=SEQ_PARENT)

    # split training set to training/val sets
    num_val = int(np.round(0.2*len(full_set)))
    num_train = len(full_set) - num_val
    train_set, val_set = random_split(full_set,[num_train, num_val])
    # tiny_set,val_set, remainder = random_split(full_set,[0.1,0.1,0.8])

    # generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # loader_tiny = DataLoader(tiny_set, batch_size=6, shuffle=True)s

    # Load Model

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
                                num_outputs=2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device in Use: %s" % device)
    model = model.to(device)

    # Setup Training

    # set training parameters
    NUM_EPOCHS = 20
    LR = 1e-4

    num_iters = len(loader_train) * NUM_EPOCHS
    warmup_period = len(loader_train) # half an epoch

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.05)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(loader_train)*2, T_mult=2)
    warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)

    network_trainer = Trainer(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            warmup_scheduler=warmup_scheduler,
                            loader_train=loader_train,
                            loader_val=loader_val,
                            loss_scale=1e1,
                            device=device,
                            check_accuracy=True,
                            verbose=len(loader_train),
                            save_weights=WEIGHT_SAVE_PATH)

    print("Training ...")
    print()
    start_time = time.time()
    outs = network_trainer.train(loss_func=nn.L1Loss(), epochs=NUM_EPOCHS)
    print("Training Time: %s" % (time.time() - start_time))
    print()

    return outs

#%%
# Call main and Visualize training
if __name__ == "__main__":
    outs = main()
    
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

    L = len(outs["rmse_train_history"])
    train_rmse = np.concatenate(outs["rmse_train_history"]).reshape(L,2)
    val_rmse = np.concatenate(outs["rmse_val_history"]).reshape(L,2)
    hitch_rmse_train = train_rmse[:,0]
    hitch_rmse_val = val_rmse[:,0]
    hr_rmse_train = train_rmse[:,1]
    hr_rmse_val = val_rmse[:,1]
    plt.subplot(2,1,1)
    plt.plot(hitch_rmse_train, '-o')
    plt.plot(hitch_rmse_val, '-o')
    plt.legend(['train', 'val'], loc='upper right')
    plt.ylabel('State1 RMSE [deg]')
    plt.xlabel('Epochs')
    plt.subplot(2,1,2)
    plt.plot(hr_rmse_train, '-o')
    plt.plot(hr_rmse_val, '-o')
    plt.ylabel('State2 RMSE [deg/s]')
    plt.xlabel('Epochs')
    plt.tight_layout()
    plt.show()