'''
############ Optuna Optimization Test ############

            FOR TEST DRIVEN DEVELOPMENT

################################################
'''
#%%
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.trainer import Trainer

NET = "vit_b_16"
SAVE = {"cond": True, "path": 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\optuna_wide\\from_scratch\\' + NET + '_optuna_studies_wide.csv'}
# WIDE_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\optuna_wide_v2\\' + NET + '_optuna_studies_wide_v2.csv'
TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v1.csv'
DEVICE = torch.device('cuda:0')

print("Optimizing %s" % NET)

# extract wide study csv and convert to dataframe
# study_df = pd.read_csv(WIDE_CSV)
# min_idx = study_df.loc[:,"value"].idxmin()
# print("From wide study: LR = %f, LS= %f" % (study_df.loc[min_idx,"params_lr"],study_df.loc[min_idx,"params_LOSS_SCALE"]))

# rmse helper function
def rmse(x_true, x_pred):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return np.sqrt(np.mean((x_true - x_pred)**2))

#%%
# Set up modified network
def define_model():
    # model = torchvision.models.resnet34(weights=False)
    # # freeze layers for fixed-feature extraction
    # # for param in model.parameters():
    # #     param.requires_grad = False

    # # modify network head
    # num_features = model.fc.in_features
    # # num_features = model.classifier.in_features
    # model.fc = nn.Sequential(
    #     nn.Linear(num_features, 500),
    #     nn.ReLU(),
    #     nn.Dropout(0.0),
    #     nn.Linear(500,300),
    #     nn.ReLU(),
    #     nn.Dropout(0.0),
    #     nn.Linear(300,1),
    # )

    model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1",
                                    # image_size=512,
                                    # patch_size=16,
                                    )
    hidden_dim = model.hidden_dim
    num_features = model.num_classes
    model.heads = nn.Sequential(     # samll head
    nn.Linear(hidden_dim,1)
    )

    if torch.cuda.is_available():
        device = DEVICE
    else:
        device = torch.device('cpu')
    print("Device in Use: %s" % device)
    torch.cuda.empty_cache()
    model = model.to(device)

    return model

#%%
# Setup data
def generate_data(TRAIN_CSV):
    # OUT_PATH = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\comp6650_proj\\mobilenetv3_small_weights2.pth'
    # TRAIN_SPECS = 'C:\\Users\\pzt0029\\Documents\\Classes\\COMP_6650_Deep_Learning\\Project\\csv_outputs\\mobilenetv3_small_train2.csv'
    full_set = TrailerData(csv_file=TRAIN_CSV,
                        transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                    ]),
                                    output_states=9)

    # split training set to training/val sets
    NUM_VAL = 1000
    NUM_TRAIN = len(full_set) - NUM_VAL
    # train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
    baby_set, train_set, val_set = random_split(full_set,[0.2,0.7,0.1])

    BATCH_SIZE = 8
    # generate loaders
    # loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_train = DataLoader(baby_set, batch_size=BATCH_SIZE, shuffle=True)
    # loader_baby = DataLoader(baby_set, batch_size=BATCH_SIZE, shuffle=True)

    return loader_train, loader_val

#%%
# Define Optuna objective

def objective(trial: optuna.Trial):

    # call model
    model = define_model()

    # define hyperparameters and generate optimizer
    # lr = trial.suggest_float("lr", 1e-7, 1e-2, log=True)
    lr_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    loss_scale_list = [1e0, 1e1, 1e2, 1e3]
    # lr_list = [x*study_df.loc[min_idx,"params_lr"] for x in range(1,10)]
    # lr_list = [x*0.0001 for x in range(1,10)]

    lr = trial.suggest_categorical("lr", lr_list)
    LOSS_SCALE = trial.suggest_categorical("LOSS_SCALE", loss_scale_list)
    # LOSS_SCALE = study_df.loc[min_idx,"params_LOSS_SCALE"]
    # LOSS_SCALE = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # generate training data
    loader_train, loader_val = generate_data(TRAIN_CSV=TRAIN_CSV)

    # check for duplicate parameters and skip if detected
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            # return t.value
            raise optuna.exceptions.TrialPruned('Duplicate parameter set')
        
    # train model
    EPOCHS = 5
    print("Training and evaluating trial ...")
    for epoch in range(EPOCHS):
        model.train()
        for idx, (x,y) in enumerate(loader_train):
            x = x.to(device=DEVICE, dtype=torch.float32)
            y = y.to(device=DEVICE, dtype=torch.float32)

            # call model to estimate
            optimizer.zero_grad()
            est = model(x)
            if isinstance(est, torchvision.models.inception.InceptionOutputs):
                est = est[0]
            est = est.squeeze()

            # compute loss
            # loss = loss_func(**params)
            loss_fun = nn.MSELoss()
            loss = loss_fun(est, y)
            loss = LOSS_SCALE * loss

            # back prop
            loss.backward()
            optimizer.step()
        
        # validate model
        est_val_history = []
        truth_val_history = []
        rmse_val = 0.0
        model.eval()
        with torch.no_grad():
            for idx, (x,y) in enumerate(loader_val):
                x = x.to(device=DEVICE, dtype=torch.float32)
                y = y.to(device=DEVICE, dtype=torch.float32)

                est = model(x)
                if isinstance(est, torchvision.models.inception.InceptionOutputs):
                    est = est[0]

                # append histories
                est_val_history.extend(est.squeeze().cpu().numpy())
                truth_val_history.extend(y.cpu().numpy())
            
        # compute rmse over entire val set
        rmse_val = rmse(np.array(truth_val_history), np.array(est_val_history))
        rmse_val = np.rad2deg(rmse_val)

        # report rmse to optuna
        trial.report(rmse_val, epoch)

        # handle pruning
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()
    
    return rmse_val

if __name__ == "__main__":

    lr_list = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    loss_scale_list = [1e0, 1e1, 1e2, 1e3]
    # combo_count = len(lr_list) * len(loss_scale_list)
    # N_TRIALS = 12
    # UNIQUE_TRIALS = combo_count
    # study = optuna.create_study(direction="minimize")
    # lr_list = [x*study_df.loc[min_idx,"params_lr"] for x in range(1,10)]
    # lr_list = [x*0.0001 for x in range(1,10)]

    search_space = {'lr':lr_list, 'LOSS_SCALE': loss_scale_list}
    # search_space = {'lr':lr_list}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(study_name="LR/LS Wide Grid",
                                sampler=sampler,
                                direction="minimize")
    study.optimize(objective, n_trials=len(lr_list)*len(loss_scale_list), timeout=None)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # unique_count = 0
    # while UNIQUE_TRIALS > len(set(str(t.params) for t in study.trials)):
    #     # print("Number of unique trials: %d" % unique_count)
    #     study.optimize(objective, n_trials=1, timeout=None)
    #     pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    #     complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))   

    # create dataframe of study
    study_df = study.trials_dataframe()
    assert isinstance(study_df, pd.DataFrame)
    # assert study_df.shape[0] == N_TRIALS

    # drop pruned trials
    drop_list = []
    for idx in range(len(study_df.value)):
        if study_df.state[idx] == 'PRUNED':
            drop_list.append(idx)
    study_df = study_df.drop(labels=drop_list)
    study_df = study_df.reset_index(drop=True)

    # export csv of study if conditioned
    if SAVE["cond"] == True:
        study_df.to_csv(SAVE["path"])