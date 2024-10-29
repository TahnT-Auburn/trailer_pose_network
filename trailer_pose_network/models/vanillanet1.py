'''
################ Vanilla Net1 Module ################

Vanilla Net1. Basic CNN Architecture

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

##################################################
'''

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


# Vanilla Net1
class VanillaNet1(nn.Module):
    def __init__(self, params):
        super(VanillaNet1, self).__init__()

        channels, height, width = params["shape_in"]
        init_f = params["initial_filters"]
        num_fc1 = params["num_fc1"]
        num_fc2 = params["num_fc2"]
        self.dropout_rate = params["dropout_rate"]

        # convolutional layers

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        # h,w=findConv2dOutShape(height,width,self.conv1)
        self.conv1_bn = nn.BatchNorm2d(init_f)

        self.conv2 = nn.Conv2d(in_channels=init_f, out_channels=init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        h,w=findConv2dOutShape(height, width, self.conv2)
        self.conv2_bn = nn.BatchNorm2d(init_f)

        self.conv3 = nn.Conv2d(in_channels=init_f, out_channels=2*init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv3.weight)
        # h,w=findConv2dOutShape(h, w, self.conv3)
        self.conv3_bn = nn.BatchNorm2d(2*init_f)

        self.conv4 = nn.Conv2d(in_channels=2*init_f, out_channels=2*init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv4.weight)
        h,w=findConv2dOutShape(h, w, self.conv4)
        self.conv4_bn = nn.BatchNorm2d(2*init_f)

        self.conv5 = nn.Conv2d(in_channels=2*init_f, out_channels=4*init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(4*init_f)

        self.conv6 = nn.Conv2d(in_channels=4*init_f, out_channels=4*init_f, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv6.weight)
        h,w=findConv2dOutShape(h, w, self.conv6)
        self.conv6_bn = nn.BatchNorm2d(4*init_f)

        # compute flatten size for the FC layers
        self.num_flatten = h * w * 4*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1,num_fc2)
        self.fc3 = nn.Linear(num_fc2,1)
        
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = self.conv1_bn(X)
        X = F.relu(self.conv2(X))
        X = self.conv2_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = self.conv3_bn(X)
        X = F.relu(self.conv4(X))
        X = self.conv4_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv5(X))
        X = self.conv5_bn(X)
        X = F.relu(self.conv6(X))
        X = self.conv6_bn(X)
        X = F.max_pool2d(X, 2, 2)
        X = F.dropout(X, self.dropout_rate)

        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        
        est = self.fc3(X)
        return est