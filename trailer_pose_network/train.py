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
from torch.utils.data import Dataset
# import model
 
from models.mango_net import mango_net

#Procedure 1 Data 
TRAIN_CSV = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/train.csv"
TEST_CSV = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/test.csv"
LEFT_IMAGE_FOLDER = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/images/LRMC/"
RIGHT_IMAGE_FOLDER = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/images/RRMC/"

tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((500,500)),
    transforms.ToTensor()
])



#Create Dataset:
class pytorch_data(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.column_names = self.df.columns
    def __len__(self):
        return len(self.df)
      
    def __getitem__(self, idx):
        #print(idx)
        #Read in 2 Images
        left_image = cv2.imread(LEFT_IMAGE_FOLDER+str(self.df[self.column_names[1]][idx])+".jpg")
        right_image = cv2.imread(RIGHT_IMAGE_FOLDER+str(self.df[self.column_names[2]][idx])+".jpg")
        #Concatenate the Two Images
        input_img = cv2.hconcat([right_image, left_image])
        input_img = tf(input_img)

        truth = self.df[self.column_names[11]][idx]
        return input_img, truth

train_dataset = pytorch_data(TRAIN_CSV)
test_dataset = pytorch_data(TEST_CSV)



model_params={
        "shape_in": (3,500,500), 
        "initial_filters": 32,    
        "num_fc1": 300,
        "dropout_rate": 0.25}

model = mango_net(model_params)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function. """
    return (torch.log(var) + ((y - mean).pow(2))/var).sum()    





opt = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.L1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=0),


loader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle = True )

loss_fun = nn.MSELoss()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(loader):
        # get the inputs; data is a list of [inputs, labels]
        
        inputs, labels = data
        inputs = inputs.to(device)
        # print(labels)
        labels = torch.tensor(labels)
        labels = labels.to(device)
        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        # mu, sig= model(inputs)
        # loss = NLLloss(labels, mu, sig)

        mu = model(inputs)
        loss = loss_fun(mu, labels.float())



        loss.backward()
        opt.step()
        

    print(loss)

torch.save(model.state_dict(), "weights/mango_net.pth")

print('Finished Training')
































