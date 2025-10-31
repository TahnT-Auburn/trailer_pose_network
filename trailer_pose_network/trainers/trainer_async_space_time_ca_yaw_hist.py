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
import copy
import keyboard

from trailer_pose_network.gradient_hooks import (
    hook_forward,
    hook_backward,
    get_all_layers,
    get_grads,
)

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
                       run_val:bool=False,
                       overfit_detector:bool=False,
                       loss_scale=1,
                       loss_save_interval:int=None,
                       device=torch.device('cpu'),
                       check_accuracy:bool=False,
                       check_gradients:bool=False,
                       verbose:int=100,
                       save_weights:str=None,
                       save_outs:str=None):

        self.model = model
        self.optimizer = optimizer
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.run_val = run_val
        self.overfit_detector = overfit_detector
        self.loss_scale = loss_scale
        self.loss_save_interval = loss_save_interval
        self.device = device
        self.check_accuracy = check_accuracy
        self.check_gradients = check_gradients
        self.verbose = verbose
        self.save_weights = save_weights
        self.save_outs = save_outs

        # apply input assertions
        if save_weights is not None and check_accuracy is False:
            raise Exception("save_weights path given but check_accuracy flag is None. Weights are saved by validation accuracy checkpoints. check_accuray must be set to True.")
        if save_weights is not None and run_val is False:
            raise Exception("save_weights path given but run_val flag is None. Weights are saved by validation accuracy checkpoints. run_val must be set to True.")
        
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
        train_loss_history = []
        train_loss1_history = []
        train_loss2_history = []
        train_loss3_history = []
        train_epoch_loss_history = []
        rmse_train_history = []
        if self.run_val:
            val_loss_history = []
            val_loss1_history = []
            val_loss2_history = []
            val_loss3_history = []
            val_epoch_loss_history = []
            rmse_val_history = []
        
        initial_loss = None

        self.model = self.model.to(device=self.device)
        
        # Set up hooks if prompted
        if self.check_gradients:
            # register hooks
            layers, grads = get_all_layers(self.model, hook_forward, hook_backward)
        
        break_outer = False
        for e in range(epochs):
            epoch_start_time = time.time()
            print("\nEpoch %d/%d" % ((e+1),epochs))
            print('-----')
            # start_time = time.time()
            
            # Press 'esc' to break loop
            if keyboard.is_pressed('esc'):
                print('"esc" detected. Exiting training.')
                break
            # maunual training termination
            if break_outer:
                break
            
            #------------------------TRAINING LOOP--------------------------
            if self.check_accuracy: # initialize/clear estimates and truth list for each epoch
                train_est_history = []
                train_truth_history = []
                val_est_history = []
                val_truth_history = []
                
            for t, (x,y) in enumerate(tqdm(self.loader_train)):
                # Press 'esc' to break loop
                if keyboard.is_pressed('esc'):
                    print('"esc" detected. Exiting training.')
                    break_outer = True
                    break
                self.model.train()

                x[0] = x[0].to(device=self.device, dtype=torch.float32) # images
                x[1] = x[1].to(device=self.device, dtype=torch.float32) # IMU
                x[2] = x[2].to(device=self.device, dtype=torch.float32) # yaw history

                y = y.to(device=self.device, dtype=torch.float32)

                # clear grads
                if self.check_gradients:
                    grads.clear()
                
                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()
                
                # call model to estimate
                est = self.model(x)
                
                # get learning rate
                lr = getLR(optimizer=self.optimizer)
                lr_history.append(lr)

                loss1 = loss_func[0](est[0], y[:,0:2])
                loss2 = loss_func[1](est[1], y[:,2:3])
                loss3 = loss_func[2](est[2], y[:,3:])
                
                loss = self.loss_scale[0] * loss1 + self.loss_scale[1] * loss2 + self.loss_scale[2] * loss3

                # perform backward pass
                loss.backward()
                
                # Clip gradient
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # update the model using the computed gradients
                self.optimizer.step()
                
                # log loss to history (if save interval not given then all losses are saved which can be expensive)
                if self.loss_save_interval is None:
                    train_loss1_history.append(loss1.detach())
                    train_loss2_history.append(loss2.detach())
                    train_loss3_history.append(loss3.detach())
                    train_loss_history.append(loss.detach())
                elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
                    train_loss1_history.append(loss1.detach())
                    train_loss2_history.append(loss2.detach())
                    train_loss3_history.append(loss3.detach())
                    train_loss_history.append(loss.detach())

                # step warmup scheduler and regular scheduler given specified order and delays
                # TODO: Make this more robust. Currently assumes scheduler steps per iteration.
                #       Works for cosine annealing schedulers but not step schedulers. Could make it an input.
                #-------------------------------------------
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
                    
                # Append estimates and truths if using for accuracy checking
                if self.check_accuracy:
                    est =  torch.cat((est[0], est[1], est[2]), dim=1)
                    train_est_history.append(est.detach())
                    train_truth_history.append(y.detach())  
                    
            # append per epoch loss history at end of training loop
            train_epoch_loss_history.append(loss.detach())
            
            # verbose updates
            if self.verbose is not None:
                tqdm.write('Epoch %d, train loss = %.4f' % (e+1,loss.detach().item()))
            #------------------------END OF TRAINING LOOP--------------------------
                
            #------------------------VALIDATION LOOP--------------------------
            if self.run_val:
                with torch.no_grad():
                    self.model.eval()
                    for t, (x,y) in enumerate(tqdm(self.loader_val)):
                    # Press 'esc' to break loop
                        if keyboard.is_pressed('esc'):
                            print('"esc" detected. Exiting training.')
                            break_outer = True
                            break

                        x[0] = x[0].to(device=self.device, dtype=torch.float32) # images
                        x[1] = x[1].to(device=self.device, dtype=torch.float32) # IMU
                        x[2] = x[2].to(device=self.device, dtype=torch.float32) # yaw history

                        y = y.to(device=self.device, dtype=torch.float32)

                        # call model to estimate
                        est = self.model(x)

                        loss1 = loss_func[0](est[0], y[:,0:2])
                        loss2 = loss_func[1](est[1], y[:,2:3])
                        loss3 = loss_func[2](est[2], y[:,3:])
                        
                        loss = self.loss_scale[0] * loss1 + self.loss_scale[1] * loss2 + self.loss_scale[2] * loss3

                        # log loss to history (if save interval not given then all losses are saved which can be expensive)
                        if self.loss_save_interval is None:
                            val_loss1_history.append(loss1.detach())
                            val_loss2_history.append(loss2.detach())
                            val_loss3_history.append(loss3.detach())
                            val_loss_history.append(loss.detach())
                        elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
                            val_loss1_history.append(loss1.detach())
                            val_loss2_history.append(loss2.detach())
                            val_loss3_history.append(loss3.detach())
                            val_loss_history.append(loss.detach())

                        # check the training and validation accuracies at the end of every epoch
                        if self.check_accuracy:
                            est =  torch.cat((est[0], est[1], est[2]), dim=1)
                            val_est_history.append(est)
                            val_truth_history.append(y)
                            
                    # append loss history per epoch
                    val_epoch_loss_history.append(loss.detach())
            
                # verbose updates
                if self.verbose is not None:
                    tqdm.write('Epoch %d, val loss = %.4f' % (e+1,loss.detach().item()))
            #------------------------------ END OF VAL LOOP --------------------------
            
            
            #--------------- GET ACCURACIES, EARLY STOPPING, OVERFIT DETECTOR, AND SAVE WEIGHTS ---------------
            if self.check_accuracy:
                rmse_train = checkAccuracy(train_est_history, train_truth_history)
                tqdm.write(f"Training RMSE: {rmse_train}")
                rmse_train_history.append(rmse_train)
                if self.run_val:
                    rmse_val = checkAccuracy(val_est_history, val_truth_history)
                    tqdm.write(f"Validation RMSE: {rmse_val}")
                    rmse_val_history.append(rmse_val)
            
                # Overfit detector
                # TODO: Make configurable and more robust
                if self.overfit_detector:
                    if np.linalg.norm(rmse_val) > np.linalg.norm(rmse_train) * 1.5:
                        tqdm.write("Overfitting detected. Terminating Training")
                        break
                
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
            
            # TODO: Add early stopping logic
            
            # -------------------- CLEAN EPOCH ------------------------
            # Delete variables for memory management
            del loss, loss1, loss2, est
            if self.check_accuracy:
                del  rmse_train
                if self.run_val:
                    del rmse_val
                 
            tqdm.write(f"Time per epoch: {time.time()-epoch_start_time}\n")
            # -------------------- END EPOCH ------------------------
        
        print("\nTraining Complete\n")

        # convert loss list to cpu
        train_loss_history = torch.stack(train_loss_history).cpu().numpy()
        train_epoch_loss_history = torch.stack(train_epoch_loss_history).cpu().numpy() if epochs>1 else train_epoch_loss_history[0].item()
        train_loss1_history = torch.stack(train_loss1_history).cpu().numpy()
        train_loss2_history = torch.stack(train_loss2_history).cpu().numpy()
        train_loss3_history = torch.stack(train_loss3_history).cpu().numpy()

        if self.run_val:
            val_loss_history = torch.stack(val_loss_history).cpu().numpy()
            val_epoch_loss_history = torch.stack(val_epoch_loss_history).cpu().numpy() if epochs>1 else val_epoch_loss_history[0].item()
            val_loss1_history = torch.stack(val_loss1_history).cpu().numpy()
            val_loss2_history = torch.stack(val_loss2_history).cpu().numpy()
            val_loss3_history = torch.stack(val_loss3_history).cpu().numpy()
        
        if self.check_gradients:
            layer_idx, avg_grads = get_grads(grads)
            
        # package outs dict for returns
        outs = {"lr_history": lr_history,
                "train_loss_history": train_loss_history,
                "train_loss1_history": train_loss1_history,
                "train_loss2_history": train_loss2_history,
                "train_loss3_history": train_loss3_history,
                "train_epoch_loss_history": train_epoch_loss_history,
            }
        if self.check_accuracy:
            outs["rmse_train_history"] = rmse_train_history
        if self.run_val:
            outs["val_loss_history"] = val_loss_history
            outs["val_loss1_history"] = val_loss1_history
            outs["val_loss2_history"] = val_loss2_history
            outs["val_loss3_history"] = val_loss3_history
            outs["val_epoch_loss_history"] = val_epoch_loss_history
            if self.check_accuracy:
                outs["rmse_val_history"] = rmse_val_history
        if self.check_gradients:
            outs["layer_idx"] = layer_idx
            outs["avg_grads"] = avg_grads
               
        if self.save_outs is not None:    
            # df = pd.DataFrame(outs)
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in outs.items()]))
            df.to_csv(self.save_outs["save_path"])
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

    # convert delta yaw pred to degrees (NOTE: Hacky)
    rmse[2] = np.rad2deg(rmse[2])

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
    