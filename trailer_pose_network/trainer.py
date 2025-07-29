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
import torchvision
import pytorch_warmup as warmup

import numpy as np
import pandas as pd
import time
from tqdm import tqdm

class Trainer():
    '''
    Trainer utility class\\
    TODO: Make this be able to handle different multi outputs, multiple heads, etc. better!
    Arguments:
        model (torch.nn.Module): 
            Neural network model
        optimizer (torch.optim): 
            Optimizer
        loader_training (torch.utils.data.DataLoader):
            Training loader
        loader_val (torch.utils.data.DataLoader): 
            Validation loader
        warmup (pytorch_warmup, optional):
            Learning rate warmup handler. Default is None.
        loss_scale (int, float, optional):
            Value to scale the loss. Default is 1 (no scaling).
        device (torch.device): 
            Device. Default is 'cpu'.
        check_accuracy (bool, optional):
            Flag to check training and validation accuracies. Default is None.\\
            NOTE: Greatly increases training time.\\
            TODO: Make it an optional to only check for training or val accuracies.
        verbose (int, optional): 
            Integer interval for printed updates.
        save_weights (str, optional): 
            Path to save weights. Default is None.
            TODO: Make this save at checkpoints instead of after a set epoch.
        save_outs (str, optional):
            Path to save training statistics (train/val RMSE, loss, learning rate). Default is None.
    '''
    def __init__(self, model,
                       optimizer,
                       loader_train,
                       loader_val,
                       warmup_scheduler=None,
                       scheduler=None,
                       loss_scale=1,
                       device=torch.device('cpu'),
                       check_accuracy:bool=False,
                       verbose:int=100,
                       save_weights:str=None,
                       save_outs:str=None):

        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.loss_scale = loss_scale
        self.device = device
        self.check_accuracy = check_accuracy
        self.verbose = verbose
        self.save_weights = save_weights
        self.save_outs = save_outs

        # apply input assertions
        if save_weights is not None and check_accuracy is False:
            raise Exception("save_weights path given but check_accuracy flag is None. Weights are saved by validation accuracy checkpoints. check_accuray must be set to True.")
        
    def train(self, loss_func, params=None, epochs=1):
        '''
        Train function to call.

        Parameters:
            loss_func (torch.nn.loss_func): Loss function
            params (dict): Dictionary of required parameters for the loss function

        Returns:
        Loss History, Training Accuracy History, and Validation Accuracy History (over epochs)
        '''

        # initilize histories
        lr_history = []
        raw_loss_history = []
        loss_history = []
        rmse_train_history = []
        rmse_val_history = []
        std_train_history = []
        std_val_history = []
        initial_loss = None

        self.model = self.model.to(device=self.device)

        for e in range(epochs):
            epoch_start_time = time.time()
            print("Epoch %d/%d" % ((e+1),epochs))
            print('-----')
            # start_time = time.time()
            for t, (x,y) in enumerate(tqdm(self.loader_train)):
                # print(f"loop restart time: {time.time()-start_time}")
                # start_time = time.time()
                # start_time = time.time()
                self.model.train()
                # NOTE: x are the images
                #       y are the estimates
                if isinstance(x,list):
                    if len(x) == 2:
                        x[0] = x[0].to(device=self.device, dtype=torch.float32)
                        x[1] = x[1].to(device=self.device, dtype=torch.float32)
                    else:
                        x = x[0] # grab first TODO: Modify this to be more interactive. Make num_inputs a parameter
                        x = x.to(device=self.device, dtype=torch.float32)
                else:
                    x = x.to(device=self.device, dtype=torch.float32)

                y = y.to(device=self.device, dtype=torch.float32)

                # call model to estimate
                est = self.model(x)
                if isinstance(est, torchvision.models.inception.InceptionOutputs):
                    est = est[0]
                # est = est.squeeze()
                # est.requires_grad_()

                # get learning rate
                lr = getLR(optimizer=self.optimizer)
                lr_history.append(lr)

                # compute loss
                # loss = loss_func(**params)
                # loss_fun = nn.L1Loss()
                # loss_fun = manual_mse_loss
                loss = loss_func(est, y)
                loss = self.loss_scale * loss
                # sum loss across states
                # loss = loss.sum()
                # if initial_loss is None:
                #     initial_loss = loss.item()
                #     loss_history.append(initial_loss)
                raw_loss_history.append(loss.detach())

                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()
                # perform backward pass
                loss.backward()
                # update the model using the computed gradients
                self.optimizer.step()

                # verbose updates
                if self.verbose is not None and t % self.verbose == 0:
                    tqdm.write('Iteration %d, loss = %.4f' % (t,loss.detach().item()))

                # step warmup scheduler and regular scheduler given specified order and delays
                # TODO: Make this more robust. Currently assumes scheduler steps per iteration.
                #       Works for cosine annealing schedulers but not step schedulers. Could make it an input.

                # if both are given
                if self.warmup_scheduler is not None and self.scheduler is not None:
                    # default is to delay the learning rate scheduler by the warmup period
                    warmup_period = self.warmup_scheduler.warmup_params[0]['warmup_period']
                    with self.warmup_scheduler.dampening():
                        if self.warmup_scheduler.last_step + 1 >= warmup_period:
                            self.scheduler.step()
                # only warmup is given
                elif self.warmup_scheduler is not None and self.scheduler is None:
                    with self.warmup_scheduler.dampening():
                        pass
                # only scheduler is given
                elif self.warmup_scheduler is None and self.scheduler is not None:
                    self.scheduler.step()
                # print(f"Time per iteration: {time.time()-start_time}\n")
                # start_time = time.time()

                

            # append histories per epoch
            loss_history.append(loss.detach())
        
            # print(f"Time per epoch: {time.time()-start_time}")

            # check the training and validation accuracies at the end of every epoch
            if self.check_accuracy:
                # start_time = time.time()
                rmse_train, rmse_val = self.checkAccuracy()
                rmse_train_history.append(rmse_train)
                rmse_val_history.append(rmse_val)

                # tqdm.write(f"Accuracy Check time: {time.time() - start_time}")
                tqdm.write(f"Training RMSE: {rmse_train}")
                tqdm.write(f"Validation RMSE: {rmse_val}")
                print()

                # save weights at lowest val RMSE checkpoint
                if self.save_weights is not None:
                    if e == 0: # initialize best rmse to first rmse
                        best_val_rmse = rmse_val
                        tqdm.write("Checkpoint reached, saving weights ...")
                        torch.save(self.model.state_dict(), self.save_weights)
                        tqdm.write("Weights saved to: %s" % self.save_weights)
                    else:
                        if np.linalg.norm(rmse_val) < np.linalg.norm(best_val_rmse):
                            best_val_rmse = rmse_val
                            tqdm.write("Checkpoint reached, saving weights ...")
                            torch.save(self.model.state_dict(), self.save_weights)
                            tqdm.write("Weights saved to: %s" % self.save_weights)

                # early stopping 
                # TODO: Change to adapt to multiple inputs and thresholds
                # TODO: Integrate an overfit catcher and deploy early stopping
                # if np.round(rmse_val[0]) <= 1 and np.round(rmse_val[1]) <= 1:
                # if np.round(rmse_val) <= 1:
                #     tqdm.write("Early Stopping Criteria Met. Terminating Training")
                #     break
                if rmse_val[0] <= 0.001 and rmse_val[1] <= 0.001:
                    tqdm.write("Early Stopping Criteria Met. Terminating Training")
                    break
                
                # early stopping due to overfitting
                # TODO: Hard set early stopping criteria, make a param?
                if np.linalg.norm(rmse_val) > np.linalg.norm(rmse_train) * 1.5:
                    tqdm.write("Overfitting detected. Terminating Training")
                    break
                
            tqdm.write(f"Time per epoch: {time.time()-epoch_start_time}")

        print("Training Complete")

        # convert loss list to cpu
        raw_loss_history = torch.stack(raw_loss_history).cpu().numpy()
        loss_history = torch.stack(loss_history).cpu().numpy() if epochs>1 else loss_history[0].item()


        # package outs dict for returns
        if self.check_accuracy:
            outs = {"lr_history": lr_history,
                    "raw_loss_history": raw_loss_history,
                    "loss_history": loss_history,
                    "rmse_train_history": rmse_train_history,
                    "rmse_val_history": rmse_val_history,}
        else:
            outs = {"lr_history": lr_history,
                    "raw_loss_history": raw_loss_history,
                    "loss_history": loss_history,}
            
        if self.save_outs is not None:    
            # df = pd.DataFrame(outs)
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in outs.items()]))
            df.to_csv(self.save_outs["save_path"])
            print("Training outs saved to: %s" % self.save_outs)

        return outs
    
    def checkAccuracy(self):
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
            # num_batches = 0   
            for train_pair,val_pair in tqdm(zip(self.loader_train,self.loader_val), total=len(self.loader_val)):
                if isinstance(train_pair[0],list):
                    if len(train_pair[0]) == 2:
                        train_pair[0][0] = train_pair[0][0].to(device=self.device, dtype=torch.float32)
                        train_pair[0][1] = train_pair[0][1].to(device=self.device, dtype=torch.float32)
                    else:
                        train_pair[0] = train_pair[0][0].to(device=self.device, dtype=torch.float32)
                    
                    x_train = train_pair[0]
                    y_train = train_pair[1].to(device=self.device, dtype=torch.float32)

                if isinstance(val_pair[0],list):
                    if len(val_pair[0]) == 2:
                        val_pair[0][0] = val_pair[0][0].to(device=self.device, dtype=torch.float32)
                        val_pair[0][1] = val_pair[0][1].to(device=self.device, dtype=torch.float32)
                    else:
                        val_pair[0] = val_pair[0][0].to(device=self.device, dtype=torch.float32)

                    x_val = val_pair[0]
                    y_val= val_pair[1].to(device=self.device, dtype=torch.float32)

                else:
                    # siphon pairs and perform device and dtype conversion
                    x_train = train_pair[0].to(device=self.device, dtype=torch.float32)
                    y_train = train_pair[1].to(device=self.device, dtype=torch.float32)

                    x_val = val_pair[0].to(device=self.device, dtype=torch.float32)
                    y_val = val_pair[1].to(device=self.device, dtype=torch.float32)

                # call model, compute estimates
                est_train = self.model(x_train)
                est_val = self.model(x_val)
                
                if isinstance(est_train, torchvision.models.inception.InceptionOutputs):
                    est_train = est_train[0]

                if isinstance(est_val, torchvision.models.inception.InceptionOutputs):
                    est_val = est_val[0]

                # applend estimations and truths for error analysis
                est_train_history.append(est_train)
                est_val_history.append(est_val)
                truth_train_history.append(y_train)
                truth_val_history.append(y_val)

            # concat history and convert to numpy
            est_train_history = torch.cat(est_train_history, dim=0)
            est_val_history = torch.cat(est_val_history, dim=0)
            truth_train_history = torch.cat(truth_train_history, dim=0)
            truth_val_history = torch.cat(truth_val_history, dim=0)

            # compute rmse
            rmse_train = rmse(truth_train_history, est_train_history)
            rmse_val = rmse(truth_val_history, est_val_history)

            # convert to numpy
            rmse_train = rmse_train.cpu().numpy().squeeze()
            rmse_val = rmse_val.cpu().numpy().squeeze()

            # convert to degrees
            # rmse_train = np.rad2deg(rmse_train)
            # rmse_val = np.rad2deg(rmse_val)
            # rmse_train[[2,3,4]] = np.rad2deg(rmse_train[[2,3,4]])
            # rmse_val[[2,3,4]] = np.rad2deg(rmse_val[[2,3,4]])

        return rmse_train, rmse_val,
    


def getLR(optimizer):
    '''
    Gets current all learning rate values from optimizer.
    '''
    # for param
    return optimizer.param_groups[0]['lr']

def rmse(x_true, x_pred, dim=0):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return torch.sqrt(torch.mean((x_true - x_pred)**2, dim=dim))

def manual_mse_loss(est, truth, dim=0):
    '''
    A manual implementation of the the MSELoss to specify the
    dimension to take the mean along
    '''
    return ((est - truth)**2).mean(dim=dim)