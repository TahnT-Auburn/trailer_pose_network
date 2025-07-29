'''
################  Test Set Testing Script ################
#
#               FOR TEST DRIVEN DEVELOPEMENT
#
##########################################################
'''
#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

from trailer_pose_network.custom_transforms import *
from trailer_pose_network.models.misc.mangonet import MangoNet
from trailer_pose_network.models.misc.vanillanet1 import VanillaNet1
from trailer_pose_network.data_setup import TrailerData

#%% 
# set test type
SET = "FF"
SUBSET = "FF1"
SAVE_SINGLE_INST = False
SINGLE_INST = True
TEST_CSV = "C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\testing\\processed\\FF\\FF1\\FF1_testing.csv"

# set weight file
NET = "mobilenetv2"
WEIGHTS = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\simulation\\finetune_transfer_learning\\full\\'+NET+'_weights.pth'

#%%
# load model with trained weights

# set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Set up modified network
model = torchvision.models.mobilenet_v2(weights=False)
# num_features = model.fc.in_features # modify network head
num_features = model.classifier[1].in_features

# model.classifier = nn.Sequential(             # large head
#     nn.Linear(num_features, 1500),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(1500,750),
#     nn.ReLU(),
#     nn.Dropout(0.0),
#     nn.Linear(750,375),
#     nn.ReLU(),
#     nn.Linear(375,1)
# )

# model.classifier = nn.Sequential(     # samll head
#     nn.Linear(num_features, 1)
# )

model.classifier = nn.Sequential(             
    nn.Linear(num_features, 500),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(500,300),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(300,1)
)

model = model.to(device)
state_dict = torch.load(WEIGHTS)
model.load_state_dict(state_dict)
# model.load_state_dict(state_dict, strict=False)

#%%
# set undistortion coefficients (for experimental data)
undistort = {
    "cameraMatrices":
    [np.array([[1209.487477, 0.000000, 1047.120765],
                        [0.000000, 1210.806093, 766.413057],
                        [0.000000, 0.000000, 1.000000]]),
    np.array([[1180.212713, 0.000000, 1009.890502],
                        [0.000000, 1182.408953, 758.997736],
                        [0.000000, 0.000000, 1.000000]])
    ],
    "distCoeffs":
    [np.array([-0.225053, 0.067380, -0.000443, -0.000453, 0.000000]),
        np.array([-0.234383, 0.075105, -0.000760, 0.000797, 0.000000])]
    
}

# Evaluate single testset
if SINGLE_INST:
    single_test_set = TrailerData(csv_file=TEST_CSV,
                                    output_states_idx=2,
                                    transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((512,512)),
                                    transforms.ToTensor()]),
                                    # undistort=undistort
                                    )

    single_test = DataLoader(single_test_set, batch_size=1, shuffle=False)

    # evalulate single model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(single_test):
            start_time = time.time()

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            est = model(x)

            est_array.append(est)
            truth_array.append(y)
            
            print(f"Single loop time: {time.time()-start_time}")
            stop=1
    print("Evaluation Complete")

    # populate dataframe and update csv
    # if SAVE_SINGLE_INST:
    #     single_inst_df = pd.read_csv(SINGLE_INST)
    #     # single_inst_df = pd.DataFrame()
    #     single_inst_df[NET] = np.rad2deg(est_array)
    #     single_inst_df.to_csv(SINGLE_INST)

    # plot estimates vs truth
    # plt.subplot(2,1,1)
    # plt.plot((1/40)*np.arange(0,len(est_array)),np.rad2deg(est_array))
    # plt.plot((1/40)*np.arange(0,len(truth_array)),np.rad2deg(truth_array), '--')
    # # plt.xlabel('Frames')
    # plt.ylabel('Hitch (deg)')
    # plt.legend(['Estimated', 'Truth'], loc='upper left')
    # plt.subplot(2,1,2)
    # plt.plot((1/40)*np.arange(0,len(est_array)), np.rad2deg(truth_array) - np.rad2deg(est_array))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Error [deg]')
    # plt.tight_layout()
    # plt.show()

    plt.subplot(2,1,1)
    plt.plot(np.rad2deg(est_array))
    plt.plot(np.rad2deg(truth_array), '--')
    # plt.xlabel('Frames')
    plt.ylabel('Hitch (deg)')
    plt.legend(['Estimated', 'Truth'], loc='upper left')
    plt.subplot(2,1,2)
    plt.plot(np.rad2deg(truth_array) - np.rad2deg(est_array))
    plt.xlabel('Time (s)')
    plt.ylabel('Error [deg]')
    plt.tight_layout()
    plt.show()

# fig, ax = plt.subplots(2,1)
# est, = ax[0].plot(np.rad2deg(est_array[0]))
# truth, = ax[0].plot(np.rad2deg(truth_array[0]), '--')
# # ax[0].xlabel('Frames')
# # ax[0].ylabel('Hitch [deg]')
# ax[0].legend(['Estimated', 'Truth'], loc='upper left')
# ax[0].set(xlim=[0,len(est_array)],
#           ylim=[-20,30],
#           ylabel = 'Hitch [deg]',
#           xlabel = 'Frames')

# # plt.subplot(2,1,2)
# error, = ax[1].plot(np.rad2deg(truth_array[0]) - np.rad2deg(est_array[0]))
# # ax[1].xlabel('Frames')
# # ax[1].ylabel('Error [deg]')
# # ax[1].title('Trailer Articulation Angle')
# ax[1].set(xlim=[0,len(est_array)],
#           ylim=[-2.5,1.5],
#           ylabel = 'Error [deg]',
#           xlabel = 'Frames')

# fig.tight_layout()
# # plt.show()

# def animate(i):
#     est.set_data(np.arange(i), np.rad2deg(est_array[:i]))
#     truth.set_data(np.arange(i), np.rad2deg(truth_array[:i]))
#     error.set_data(np.arange(i), np.rad2deg(truth_array[:i]) - np.rad2deg(est_array[:i]))
#     return est,truth,error

# ani = animation.FuncAnimation(fig=fig,
#                               func=animate,
#                               interval=30,
#                               frames=len(est_array))
#                             #   frames=len(est_array))

# # ani.save("single_instance.mp4")
# writergif = animation.PillowWriter(fps=40)
# ani.save('resnet34_single_instance_v2.gif', writer=writergif)

# plt.show()

#%%
# Evaluate entire testset
# load data using pytorch dataset and dataloader tools

else:

    test_set = TrailerData(csv_file=TEST_CSV,
                            output_states_idx=2,
                            transform=transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize((512,512)),
                                    transforms.ToTensor(),
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                    #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                    ]),
                            # undistort=undistort
                            )

    # generate loader
    BATCHSIZE = 24
    loader_test = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=True)

    # evalulate model
    est_array = []
    truth_array = []
    print("Evaluating Model ...")
    with torch.no_grad():
        model.eval()
        for t, (x,y) in enumerate(loader_test):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            est = model(x)

            est_array.extend(est.squeeze().cpu().numpy())
            truth_array.extend(y.cpu().numpy())
    print("Evaluation Complete")


    # Compute RMSE of testing set
    def rmse(x_true, x_pred):
        '''
        Calculates root mean squared error (RMSE)
        '''
        return np.sqrt(np.mean((x_true - x_pred)**2))

    rmse_test = rmse(np.array(truth_array), np.array(est_array))
    rmse_test = np.rad2deg(rmse_test)
    print("Test Set RMSE: %f" % rmse_test)




