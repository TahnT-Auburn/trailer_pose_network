'''
######################################################
Script to test warmp up and learning rate schedulers
######################################################
'''
#%%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import pytorch_warmup as warmup

#%%
# generate training settings
train_data_length = 2000
epochs = 5
num_iters = train_data_length * epochs
warmup_period = train_data_length//2

# basic model
model = nn.Sequential(
        nn.Linear(10,20),
        nn.GELU(),
        nn.Linear(20,10))

# set optimizer, scheduler and warmup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=warmup_period*2, T_mult=2)
warmup_scheduler = warmup.LinearWarmup(optimizer=optimizer, warmup_period=warmup_period)

#%%
def getLR(optimizer):
    '''
    Gets current all learning rate values from optimizer.
    '''
    # for param
    return optimizer.param_groups[0]['lr']

# set mock training loop
lr_history = []
for e in range(epochs):
    for i in range(0,train_data_length):
        
        # grab current learning rate
        lr = getLR(optimizer=optimizer)
        lr_history.append(lr)

        # step optimzier
        optimizer.step()

        # step scheduler
        # with warmup_scheduler.dampening():
        #     scheduler.step()
        with warmup_scheduler.dampening():
            if warmup_scheduler.last_step + 1 >= warmup_period:
                scheduler.step()

        #     if epochs >= warmup_epochs:
        #         lr_scheduler.step(epoch-warmup_epochs + i / iters)

#%%
# plot learining rate
plt.plot(lr_history)
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.show()