'''
####################### Trainer Class #######################

Trainer utility class designed to train models, and track
loss histories and training/val accuracies
TODO: Move to a Network Utilities package

Author: Tahn Thawainin, AU GAVLAB
        email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn

#############################################################
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Trainer():
    '''
    Trainer utility class

    Arguments:
        model (torch.nn.Module): Neural network model
        optimizer (torch.optim): Optimizer
        loader_training (torch.utils.data.DataLoader): Training loader
        loader_val (torch.utils.data.DataLoader): Validation loader
        device (torch.device): Device (default='cpu')
        verbose (dict): Dictionary containing verbose boolean condition
                        and print rate. \\
                        Items: {"cond": boolean, "print_every": int}
    '''
    def __init__(self, model,
                       optimizer,
                       loader_train,
                       loader_val,
                       device=torch.device('cpu'),
                       verbose={"cond": True, "print_every": 100}):

        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.device = device
        self.verbose = verbose

    def check_accuracy(self):
        '''
        Checks the accuracy of the network.

        Returns:
            Training accuracy and validation accuracy
        '''
        num_correct_train, num_samples_train, num_correct_val, num_samples_val = 0, 0, 0, 0
        with torch.no_grad():   
            for train_pair,val_pair in zip(self.loader_train,self.loader_val):
                
                # siphon pairs and perform device and dtype conversion
                x_train = train_pair[0].to(device=self.device, dtype=torch.float32)
                y_train = train_pair[1].to(device=self.device, dtype=torch.float32)

                x_val = val_pair[0].to(device=self.device, dtype=torch.float32)
                y_val = val_pair[1].to(device=self.device, dtype=torch.float32)

                # call model, compute estimates
                est_train = self.model(x_train)
                est_val = self.model(x_val)
                # _, preds_train = est_train.max(1)
                # _, preds_val = est_val.max(1)

                # evaulate train predictions for train accuracy
                num_correct_train += (est_train == y_train).sum()
                num_samples_train += est_train.size(0)

                # evaulate val predictions for val accuracy
                num_correct_val += (est_val == y_val).sum()
                num_samples_val += est_val.size(0)

            # compute accuracies   
            acc_train = float(num_correct_train) / num_samples_train
            acc_val = float(num_correct_val) / num_samples_val

        return acc_train, acc_val
    
    def train(self, loss_func=None, params=None, epochs=1):
        '''
        Train function to call.

        Parameters:
            loss_func (torch.nn.loss_func): Loss function
            params (dict): Dictionary of required parameters for the loss function

        Returns:
        Loss History, Training Accuracy History, and Validation Accuracy History (over epochs)
        '''

        # initilize histories
        loss_history = []
        acc_train_history = []
        acc_val_history = []
        initial_loss = None

        self.model = self.model.to(device=self.device)

        for e in range(epochs):
            print("Epoch %d/%d" % ((e+1),epochs))
            print('-----')
            for t, (x,y) in enumerate(self.loader_train):
                self.model.train()
                # NOTE: x are the images
                #       y is the hitch angle estimate
                x = x.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.float32)

                # call model to estimate
                est = self.model(x)

                # compute loss
                # loss = loss_func(**params)
                loss_fun = nn.MSELoss()
                loss = loss_fun(est, y)
                if initial_loss is None:
                    initial_loss = loss.item()
                    loss_history.append(initial_loss)

                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()

                # perform backward pass
                loss.backward()
                
                # update the model using the computed gradients
                self.optimizer.step()

                # verbose updates
                if self.verbose["cond"] == True and t % self.verbose["print_every"] == 0:
                    print('Iteration %d, loss = %.4f' % (t,loss.item()))

            # check the training and validation accuracies at the end of every epoch
            # acc_train, acc_val = self.check_accuracy()
            # print('Training Accuracy at Epoch %d: %.2f%%' % (e+1, 100*acc_train))
            # print('Validation Accuracy at Epoch %d: %.2f%%' % (e+1, 100*acc_val))
            # print()

            # append histories per epoch
            loss_history.append(loss.item())
            # acc_train_history.append(acc_train)
            # acc_val_history.append(acc_val)
        
        return loss_history, acc_train_history, acc_val_history