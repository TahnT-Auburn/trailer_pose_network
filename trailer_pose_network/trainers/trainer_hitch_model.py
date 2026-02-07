import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_warmup as warmup

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import copy
import keyboard

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loader_train,
        loader_val,
        warmup_scheduler=None,
        scheduler=None,
        overfit_detector:bool=False,
        loss_scale=1,
        device=torch.device('cpu'),
        verbose:int=100,
        save_weights:str=None,
        save_outs:str=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.warmup_scheduler = warmup_scheduler
        self.scheduler = scheduler
        self.overfit_detector = overfit_detector
        self.loss_scale = loss_scale
        self.device = device
        self.verbose = verbose
        self.save_weights = save_weights
        self.save_outs = save_outs
        
    def train(self, loss_func, epochs):
        lr_history = []
        train_loss_history = []
        train_epoch_loss_history = []
        rmse_train_history = []
        val_loss_history = []
        val_epoch_loss_history = []
        rmse_val_history = []

        self.model = self.model.to(device=self.device)
        
        for e in range(epochs):
            epoch_start_time = time.time()
            print("\nEpoch %d/%d" % ((e+1),epochs))
            print('-----')
            
            # accumulators for accuracy
            train_est_history = []
            train_truth_history = []
            val_est_history = []
            val_truth_history = []

            # ---- Training Loop ----
            for t, (x,y) in enumerate(tqdm(self.loader_train)):
                # x is images and y are hitch targets
                self.model.train()
                x = x.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.float32)
                y = y.unsqueeze(1) # TODO: Fix
                
                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()
                
                # get learning rate
                lr = getLR(optimizer=self.optimizer)
                lr_history.append(lr)
                
                # call model
                hitch_est = self.model(x)

                loss = loss_func(hitch_est, y)
                loss = self.loss_scale*loss
                
                loss.backward()
                self.optimizer.step()
                
                # populate accumulators
                train_loss_history.append(loss.detach())
                train_est_history.append(hitch_est.detach())
                train_truth_history.append(y.detach())                
                
                # warmup and scheduler LR dynamics
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
                
            # append per epoch loss history at end of training loop
            train_epoch_loss_history.append(loss.detach())
            
            # verbose updates
            if self.verbose is not None:
                tqdm.write('Epoch %d, train loss = %.4f' % (e+1,loss.detach().item()))
                
            # ---- Val Loop ----
            with torch.no_grad():
                self.model.eval()
                for t, (x,y) in enumerate(tqdm(self.loader_val)): 
                    x = x.to(device=self.device, dtype=torch.float32)
                    y = y.to(device=self.device, dtype=torch.float32)
                    y = y.unsqueeze(1) # TODO: Fix
                    
                    # call model
                    hitch_est = self.model(x)

                    loss = loss_func(hitch_est, y)
                    loss = self.loss_scale*loss

                    # populate accumulators
                    val_loss_history.append(loss)
                    val_est_history.append(hitch_est.detach())
                    val_truth_history.append(y.detach())
                    
                # append loss history per epoch
                val_epoch_loss_history.append(loss.detach())
                
            # verbose updates
            if self.verbose is not None:
                tqdm.write('Epoch %d, val loss = %.4f' % (e+1,loss.detach().item()))

            #----- GET ACCURACIES, EARLY STOPPING, OVERFIT DETECTOR, AND SAVE WEIGHTS ------
            # training RMSE
            rmse_train = checkAccuracy(train_est_history, train_truth_history)
            tqdm.write(f"Training RMSE: {rmse_train}")
            rmse_train_history.append(rmse_train) 
            
            # val RMSE
            rmse_val = checkAccuracy(val_est_history, val_truth_history)
            tqdm.write(f"Validation RMSE: {rmse_val}")
            rmse_val_history.append(rmse_val)
            
            # Save weights based on best validation RMSE
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
                        
            # -------------------- CLEAN EPOCH ------------------------
            del loss, hitch_est, rmse_train, rmse_val
            
            tqdm.write(f"Time per epoch: {time.time()-epoch_start_time}\n")
            
            # -------------------- END EPOCH ------------------------
        print("\nTraining Complete\n")            
        
        # bookeeping for log
        train_loss_history = torch.stack(train_loss_history).cpu().numpy()
        train_epoch_loss_history = torch.stack(train_epoch_loss_history).cpu().numpy() if epochs>1 else train_epoch_loss_history[0].item()
        val_loss_history = torch.stack(val_loss_history).cpu().numpy()
        val_epoch_loss_history = torch.stack(val_epoch_loss_history).cpu().numpy() if epochs>1 else val_epoch_loss_history[0].item()
        
        # package outs dict for returns
        outs = {"lr_history": lr_history,
                "train_loss_history": train_loss_history,
                "train_epoch_loss_history": train_epoch_loss_history,
                "rmse_train_history":  rmse_train_history,
                "val_loss_history": val_loss_history,
                "val_epoch_loss_history": val_epoch_loss_history,
                "rmse_val_history": rmse_val_history,
            }
        # save training log
        if self.save_outs is not None:    
            # df = pd.DataFrame(outs)
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in outs.items()]))
            df.to_csv(self.save_outs)
            print("Training outs saved to: %s" % self.save_outs)
            
            return self.model, outs
                 
def checkAccuracy(est_history, truth_history):
    '''
    Returns the RMSE between an estimated and truth histories. Designed
    to accept a non-uniform list of batched tensors.
    Args:
        est_history (list): Estimated history
        truth_history (list): Truth history
    Returns:
        rmse_val (float): RMSE
    '''
    # concat history and convert to numpy
    est_history = torch.cat(est_history, dim=0)
    truth_history = torch.cat(truth_history, dim=0)

    # compute rmse
    rmse = get_rmse(truth_history, est_history)

    # convert to numpy
    rmse = rmse.cpu().numpy().squeeze()

    # convert hitch rmse to degrees (NOTE: Hacky)
    rmse = np.rad2deg(rmse)

    # take mean across temporal dim
    # rmse = np.mean(rmse, axis=1)
    
    return rmse
    

def getLR(optimizer):
    '''
    Gets current all learning rate values from optimizer.
    '''
    # for param
    return optimizer.param_groups[0]['lr']

def get_rmse(x_true, x_pred, dim=0):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return torch.sqrt(torch.mean((x_true - x_pred)**2, dim=dim))