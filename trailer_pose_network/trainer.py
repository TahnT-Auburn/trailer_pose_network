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
        save_weights (dict): Dictionary containing a save weight boolean condition
                             and output path string. \\
                             Items: {"cond": boolean, "save_path": string}
    '''
    def __init__(self, model,
                       optimizer,
                       loader_train,
                       loader_val,
                       loss_scale=1,
                       device=torch.device('cpu'),
                       verbose={"cond": True, "print_every": 100},
                       save_weights={"cond": False, "save_path": None}):

        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.loss_scale = loss_scale
        self.device = device
        self.verbose = verbose
        self.save_weights = save_weights


    
    def check_accuracy(self):
        '''
        Checks the accuracy of the network.

        Returns:
            Training accuracy and validation accuracy
        '''
        
        # clear histories and reset values
        est_train_history = []
        est_val_history = []
        truth_train_history = []
        truth_val_history = []

        rmse_train = 0
        rmse_val = 0
        std_train = 0
        std_val = 0
        err_fun = nn.MSELoss()
        with torch.no_grad():
            num_batches = 0   
            for train_pair,val_pair in zip(self.loader_train,self.loader_val):
                
                # siphon pairs and perform device and dtype conversion
                x_train = train_pair[0].to(device=self.device, dtype=torch.float32)
                y_train = train_pair[1].to(device=self.device, dtype=torch.float32)

                x_val = val_pair[0].to(device=self.device, dtype=torch.float32)
                y_val = val_pair[1].to(device=self.device, dtype=torch.float32)

                # call model, compute estimates
                est_train = self.model(x_train)
                est_val = self.model(x_val)
                
                # applend estimations and truths for error analysis
                est_train_history.extend(est_train.squeeze().cpu().numpy())
                est_val_history.extend(est_val.squeeze().cpu().numpy())
                truth_train_history.extend(y_train.cpu().numpy())
                truth_val_history.extend(y_val.cpu().numpy())

                # # compute rmse
                # rmse_train += np.sqrt(err_fun(est_train.squeeze(), y_train).item())
                # rmse_val += np.sqrt(err_fun(est_val.squeeze(), y_val).item())
                
                # # compute error STD
                # std_train += np.std((y_train.cpu().numpy()- est_train.squeeze().cpu().numpy()))
                # std_val += np.std((y_val.cpu().numpy() - est_val.squeeze().cpu().numpy()))

                # update number of batches finished
                num_batches += 1

                # if num_batches == 10:
                #     break
            
            # compute rmse
            rmse_train = rmse(np.array(truth_train_history), np.array(est_train_history))
            rmse_val = rmse(np.array(truth_val_history), np.array(est_val_history))

            # compute error stds
            error_train = np.array(truth_train_history )- np.array(est_train_history)
            error_val = np.array(truth_val_history) - np.array(est_val_history)
            std_train = np.std(error_train)
            std_val = np.std(error_val)
            
            # convert to degrees
            rmse_train = np.rad2deg(rmse_train)
            rmse_val = np.rad2deg(rmse_val)
            
            std_train = np.rad2deg(std_train)
            std_val = np.rad2deg(std_val)

        return rmse_train, rmse_val, std_train, std_val
    
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
        rmse_train_history = []
        rmse_val_history = []
        std_train_history = []
        std_val_history = []
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
                est = est.squeeze()

                # compute loss
                # loss = loss_func(**params)
                loss_fun = nn.MSELoss()
                loss = loss_fun(est, y)
                loss = self.loss_scale * loss
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
            rmse_train, rmse_val, std_train, std_val = self.check_accuracy()
            print('Training RMSE | Error STD: %.2f | %.2f' % (rmse_train, std_train))
            print('Validation RMSE | Error STD: %.2f | %.2f' % (rmse_val, std_val))
            print()

            # append histories per epoch
            loss_history.append(loss.item())
            rmse_train_history.append(rmse_train)
            rmse_val_history.append(rmse_val)
            std_train_history.append(std_train)
            std_val_history.append(std_val)
        
        print("Training Complete")

        # save weights if conditioned
        if self.save_weights["cond"] == True:
            torch.save(self.model.state_dict(), self.save_weights["save_path"])
            print("Weights saved to: %s" % self.save_weights["save_path"])
        
        # package outs dict for returns
        outs = {"loss_history": loss_history,
                "rmse_train_history": rmse_train_history,
                "rmse_val_history": rmse_val_history,
                "std_train_history": rmse_val_history,
                "std_val_history": std_val_history}
        
        return outs
    

def rmse(x_true, x_pred):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return np.sqrt(np.mean((x_true - x_pred)**2))