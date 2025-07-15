'''
############ Transfer Learning With Ray Tune ############

            FOR TEST DRIVEN DEVELOPMENT

#########################################################
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler

import numpy as np
import matplotlib.pyplot as plt

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.trainer import Trainer

#%%
# Set up modified network
model = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')

# modify network head
# num_features = model.fc.in_features
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1),
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
torch.cuda.empty_cache()
model = model.to(device)

#%%
# Setup data
def load_data():
    TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v1.csv'
    # OUT_PATH = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\test_densenet121_weights.pth'
    full_set = TrailerData(csv_file=TRAIN_CSV,
                        transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((512,512)),
                                    transforms.ToTensor(),
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                    ]))

    # split training set to training/val sets
    NUM_VAL = 1000
    NUM_TRAIN = len(full_set) - NUM_VAL
    train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
    # baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

    BATCH_SIZE = 8
    # generate loaders
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return loader_train, loader_val


def rmse(x_true, x_pred):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return np.sqrt(np.mean((x_true - x_pred)**2))

def train_with_ray_tune(config, ):
    epochs=5
    loss_scale=100
    # define loss function
    criterion = nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"],
                                 betas=(config["b1"], config["b2"]),
                                 weight_decay=config["weight_decay"])
    
    # load data
    loader_train, loader_val = load_data()

    # train
    for epoch in range(epochs):
        print("Epoch %d/%d" % ((epoch+1),epochs))
        print('-----')
        model.train()
        for t, (x,y) in enumerate(loader_train,0):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            
            # estimate
            optimizer.zero_grad()
            est = model(x)

            # compute loss and backpropagate
            loss = criterion(est, y)
            loss = loss_scale * loss
            loss.backward()
            optimizer.step()
        
        # validation loss (to report to ray tune)
        val_rmse = 0.0
        truth_val_history = []
        est_val_history = []
        for t, (x,y) in enumerate(loader_val, 0):
            with torch.no_grad():
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
                est = model(x)
                est_val_history.extend(est.squeeze().cpu().numpy())
                truth_val_history.extend(y.cpu().numpy())
        
        val_rmse = rmse(np.array(truth_val_history), np.array(est_val_history))
        val_rmse = np.rad2deg(val_rmse)

        # report validation rmse to ray tune
        train.report({"val_rmse": val_rmse})
    
    print("Training Complete")

#%%
# Running cross-validation

# define hyperparameters search space
config = {
    "lr": tune.choice([1e-3, 1e-4, 1e-5, 1e-6, 1e-7]),
    "b1": tune.choice([0.9, 0.95, 0.99]),
    "b2": tune.choice([0.999, 0.9995, 0.9999]),
    "weight_decay": tune.choice([1e-3, 1e-4, 1e-5])
}

# run hyperparameter tuning
scheduler = ASHAScheduler(metric="val_rmse", mode="min", max_t=10, grace_period=1, reduction_factor=2)

result = tune.run(train_with_ray_tune,
                  resources_per_trial={"cpu":16, "gpu": 1},
                  config=config,
                  num_samples=10,
                  scheduler=scheduler)

print("Best config: ", result.get_best_config(metric="val_rmse", mode="min"))
