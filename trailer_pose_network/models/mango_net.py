import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import math




def findConv2dOutShape(hin,win,conv,pool=2):
    # get conv arguments
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)




# Neural Network
class mango_net(nn.Module):
    # Network Initialisation
    def __init__(self, params):
        super(mango_net, self).__init__()
        

        channels, height, width=params["shape_in"]
        init_f=params["initial_filters"] 
        num_fc1=params["num_fc1"]  
        self.dropout_rate=params["dropout_rate"] 
        


        
        # Convolution Layers
        self.conv1 = nn.Conv2d(channels, init_f, kernel_size=3)      #Three channel, 16 filters, 3x3 kernel 
        h,w=findConv2dOutShape(height,width,self.conv1)
        self.conv1_bn = nn.BatchNorm2d(init_f)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv2_bn = nn.BatchNorm2d(init_f*2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv3_bn = nn.BatchNorm2d(init_f*4)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        self.conv4_bn = nn.BatchNorm2d(init_f*8)
        # compute the flatten size
        self.num_flatten=h*w*8*init_f #Compute the number of nodes in output layer 
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc_mu = torch.nn.Linear(num_fc1, 1, bias=True)
        self.fc_sig = torch.nn.Linear(num_fc1, 1, bias = True)
        
        # self.fc2 = nn.Linear(num_fc1, 1)

    def forward(self,X):
        
        # Convolution & Pool Layers
        X = F.relu(self.conv1(X)); 
        # X = self.conv1_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        # X = self.conv2_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        # X = self.conv3_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        # X = self.conv4_bn(X)
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, self.num_flatten)
        
        X = F.relu(self.fc1(X))
        # X=F.dropout(X, self.dropout_rate)
        mu= self.fc_mu(X)
        # sig = self.fc_sig(X)
        # sig = F.softplus(sig)+1e-6
        return mu




































