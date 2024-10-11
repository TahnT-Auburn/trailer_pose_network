import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torch import optim
from PIL import Image
from torch.utils.data import Dataset, random_split
# import model
import math

from data_setup import TrailerData
from custom_transforms import *
from trainer_wUncertainty import TrainerwStd



# from models.mango_net import mango_net
from models.mango_net_wUncertainty import mango_net

#Procedure 1 Data 
# TRAIN_CSV = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/train.csv"
# TEST_CSV = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/test.csv"
# LEFT_IMAGE_FOLDER = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/images/LRMC/"
# RIGHT_IMAGE_FOLDER = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/images/RRMC/"

TRAIN_CSV = '/home/gavlab/Trailer_Pitch_Estimate/TrainingData/RoadRunner/TrainingData/Combined/combined_one_instance_reduced.csv'

tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

full_set = TrailerData(csv_file=TRAIN_CSV,
                       transform=tf)

#Create Dataset:
# class pytorch_data(Dataset):
#     def __init__(self, csv_file):
#         self.df = pd.read_csv(csv_file)
#         self.column_names = self.df.columns
#     def __len__(self):
#         return len(self.df)
      
#     def __getitem__(self, idx):
#         #print(idx)
#         #Read in 2 Images
#         left_image = cv2.imread(LEFT_IMAGE_FOLDER+str(self.df[self.column_names[1]][idx])+".jpg")
#         right_image = cv2.imread(RIGHT_IMAGE_FOLDER+str(self.df[self.column_names[2]][idx])+".jpg")
#         #Concatenate the Two Images
#         input_img = cv2.hconcat([right_image, left_image])
#         input_img = tf(input_img)

#         truth = self.df[self.column_names[11]][idx]
#         return input_img, truth

NUM_VAL = 1000
NUM_TRAIN = len(full_set) - NUM_VAL
# train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

# generate loaders
loader_train = DataLoader(train_set, batch_size=20, shuffle=True)
loader_val = DataLoader(val_set, batch_size=20, shuffle=True)
loader_baby = DataLoader(baby_set, batch_size=20, shuffle=True)

# train_dataset = pytorch_data(TRAIN_CSV)
# test_dataset = pytorch_data(TEST_CSV)



model_params={
        "shape_in": (3,512,512), 
        "initial_filters": 32,    
        "num_fc1": 300,
        "dropout_rate": 0.25}

model = mango_net(model_params)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device in use: ", device)

model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-7, weight_decay= 1e-3)

network_trainer = TrainerwStd(model=model,
                          optimizer=opt,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_weight = 100,
                          device=device,
                          verbose={"cond":True,"print_every":25})


loss_history, err_train_history, err_val_history, model_weights= network_trainer.train(epochs=10)

torch.save(model_weights, "mango_net_wSTD2.pth")
# plot loss
plt.subplot(2,1,1)
plt.plot(loss_history, '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plot accuracies
plt.subplot(2,1,2)
plt.plot(err_train_history, '-o')
plt.plot(err_val_history, '-o')
plt.legend(['train', 'val'], loc='lower right')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.tight_layout()


plt.savefig('loss.png')
plt.show()


print('Finished Training')
# def NLLloss(y, mean, var):
#     """ Negative log-likelihood loss function. """
#     return (torch.log(var) + ((y - mean).pow(2))/var).sum()    



# def MAE(x,y):
#     loss = math.sqrt(abs(x**2-y**2))
#     return loss



# criterion = nn.L1Loss()
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=0),


# loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle = True )

# loss_fun = nn.MSELoss()

# for epoch in range(10):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(loader):
#         # get the inputs; data is a list of [inputs, labels]
        
#         inputs, labels = data
#         inputs = inputs.to(device)
#         # print(labels)
#         labels = torch.tensor(labels)
#         labels = labels.to(device)
#         # zero the parameter gradients
#         opt.zero_grad()

#         # forward + backward + optimize
#         # mu, sig= model(inputs)
#         # loss = NLLloss(labels, mu, sig)

#         mu = model(inputs)
#         # loss = loss_fun(mu, labels.float())
#         loss = MAE(mu,labels.float())



#         loss.backward()
#         opt.step()
        

#     print(loss)

# torch.save(model.state_dict(), "weights/mango_net.pth")

# print('Finished Training')
































