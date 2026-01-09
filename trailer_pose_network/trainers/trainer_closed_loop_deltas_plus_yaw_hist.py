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
import random

from trailer_pose_network.gradient_hooks import (
    hook_forward,
    hook_backward,
    get_all_layers,
    get_grads,
)
#TODO: ADD CLOSED LOOP SCHEDULER LOGIC! To train off previous prediction ab yaws in the sequence.
#      Be sure to modify the dataloader so we get a yaw history input and target yaw outputs.
#      Once the scheduler has been reached, shift from training on GT to model predictions (closed loop!)
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
        train_total_loss_hist = []
        train_trans_loss_hist = []
        train_rot_loss_hist = []
        train_acc_yaw_loss_hist = []
        train_abs_yaw_loss_hist = []
        train_epoch_loss_history = []
        rmse_train_history = []
        if self.run_val:
            val_total_loss_hist = []
            val_trans_loss_hist = []
            val_rot_lost_hist = []
            val_acc_yaw_loss_hist = []
            val_abs_yaw_loss_hist = []
            val_epoch_loss_history = []
            rmse_val_history = []

        # cast model to device
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
                train_acc_yaw_est_hist = []
                train_truth_history = []
                train_acc_yaw_truth_hist = []
                val_est_history = []
                val_truth_history = []
                val_acc_yaw_est_hist = []
                val_acc_yaw_truth_hist = []
            # training_loop_start_time = time.time()
            print_once = False
            for t, (x,y) in enumerate(tqdm(self.loader_train)):
                # print(f'Training loop restart time: {time.time() - training_loop_start_time}')
                # Press 'esc' to break loop
                if keyboard.is_pressed('esc'):
                    print('"esc" detected. Exiting training.')
                    break_outer = True
                    break
                self.model.train()

                # extract input
                images = x[0] # [B, T_img, C_img, W, H]
                inerts = x[1] # [B, T_inert, C_inert]
                yaw_hist = x[2]
                # get values from dimensions
                B = images.shape[0]
                num_deltas = y.shape[2]
                
                # cast inputs to device
                images = images.to(device=self.device, dtype=torch.float32)
                inerts = inerts.to(device=self.device, dtype=torch.float32)
                yaw_hist = yaw_hist.to(device=self.device, dtype=torch.float32)
                
                # cast output to device
                y = y.to(device=self.device, dtype=torch.float32) #[B, 3, num_deltas]
                
                # clear grads
                if self.check_gradients:
                    grads.clear()
                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()
                
                # get learning rate for visualization
                lr = getLR(optimizer=self.optimizer)
                lr_history.append(lr)
                
                # initialize acummulated yaw
                acc_yaw_pred = torch.zeros(B, 1, device=self.device)
                acc_yaw_truth = torch.zeros(B, 1, device=self.device)
                
                # intialize other variables
                trans_losses = []
                rot_losses = []
                abs_yaw_losses = []
                # gt_probability = max(0.0, 1.0 - (e / epochs))
                gt_probability = get_gt_probability(e, epochs)
                if random.random() > gt_probability:
                    pure_closed_loop = True
                if not print_once:
                    if e > epochs//2 and pure_closed_loop:
                        tqdm.write(f'Training on pure closed-loop')
                        print_once = True
                for i in range (0,num_deltas):
                    # slice inputs to get indivual intervals 
                    image_input = images[:, i:i+2]
                    inert_input = inerts[:, i*4:i*4+5]
                    # yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                    # ==== SCHEDULED SAMPLING ====
                    if e <= epochs // 2: # for fist half of training use per step propbability
                        if i == 0: # initialize with ground truth 
                            yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                        elif random.random() < gt_probability: # if not the start of the interval, have a chance to still train from GT w/ noise
                            yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                        else: # else use the previous yaw estimate from model
                            yaw_input = yaw_est.detach().unsqueeze(dim=1)
                    else: # after half way start having a chance for pure closed loop training per-sequence
                        if pure_closed_loop:
                            if i == 0: # initialize with noisy ground truth 
                                yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                            else: # else use the previous yaw estimate from model
                                yaw_input = yaw_est.detach().unsqueeze(dim=1)
                        else: # regular per-step
                            if i == 0: # initialize with ground truth 
                                yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                            elif random.random() < gt_probability: # if not the start of the interval, have a chance to still train from GT w/ noise
                                yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                            else: # else use the previous yaw estimate from model
                                yaw_input = yaw_est.detach().unsqueeze(dim=1)
                    # construct input to model
                    inputs = [image_input, inert_input, yaw_input]
                    # slice delta output target
                    target = y[:,:,i]
                    trans_target = target[:,0:2] # [B,2] first two predictions are translations
                    rot_target = target[:,2:3] #  [B,1] middle one is rotation
                    yaw_target = target[:,3:] # [B,2] last two are sin/cos comps of abs yaw
                    # call model
                    # model_start_time = time.time()
                    est = self.model(inputs)
                    # print(f'Model time: {time.time() - model_start_time}')
                    # parse estimates
                    trans_est = est[0] # [B,2]
                    rot_est = est[1] # [B,1]
                    yaw_est = est[2] # [B,2]
                    # accumulate the yaw prediction over sequence length
                    acc_yaw_pred += rot_est
                    acc_yaw_truth += rot_target
                    # calculate delta losses
                    trans_loss = loss_func[0](trans_est, trans_target)
                    rot_loss = loss_func[1](rot_est, rot_target)
                    abs_yaw_loss = loss_func[2](yaw_est, yaw_target)
                    trans_losses.append(trans_loss)
                    rot_losses.append(rot_loss)
                    abs_yaw_losses.append(abs_yaw_loss)
                    # Append estimates and truths if using for accuracy checking
                    if self.check_accuracy:
                        est_tensor =  torch.cat((trans_est, rot_est, yaw_est), dim=1)
                        train_est_history.append(est_tensor.detach())
                        train_truth_history.append(target.detach())
                if self.check_accuracy:
                    train_acc_yaw_est_hist.append(acc_yaw_pred.detach())
                    train_acc_yaw_truth_hist.append(acc_yaw_truth.detach())
                # take average of deltas over sequence interval
                avg_trans_loss = torch.stack(trans_losses).mean()
                avg_rot_loss = torch.stack(rot_losses).mean()
                avg_yaw_loss = torch.stack(abs_yaw_losses).mean()
                # compute accumalted yaw loss
                acc_yaw_loss = loss_func[3](acc_yaw_pred, acc_yaw_truth)
                # total_loss = self.loss_scale[0] * avg_trans_loss + self.loss_scale[1] * avg_rot_loss + self.loss_scale[2] * avg_yaw_loss
                total_loss = self.loss_scale[0] * avg_trans_loss + self.loss_scale[1] * avg_rot_loss + self.loss_scale[2] * avg_yaw_loss + self.loss_scale[3] * acc_yaw_loss

                # perform backward pass
                total_loss.backward()

                # Clip gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # update the model using the computed gradients
                self.optimizer.step()
                
                # log loss to history (if save interval not given then all losses are saved which can be expensive)
                if self.loss_save_interval is None:
                    train_trans_loss_hist.append(avg_trans_loss.detach())
                    train_rot_loss_hist.append(avg_rot_loss.detach())
                    train_abs_yaw_loss_hist.append(avg_yaw_loss.detach())
                    train_acc_yaw_loss_hist.append(acc_yaw_loss.detach())
                    train_total_loss_hist.append(total_loss.detach())
                elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
                    train_trans_loss_hist.append(avg_trans_loss.detach())
                    train_rot_loss_hist.append(avg_rot_loss.detach())
                    train_abs_yaw_loss_hist.append(avg_yaw_loss.detach())
                    train_acc_yaw_loss_hist.append(acc_yaw_loss.detach())
                    train_total_loss_hist.append(total_loss.detach())

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
                # training_loop_start_time = time.time()        
            # append per epoch loss history at end of training loop
            train_epoch_loss_history.append(total_loss.detach())
            
            # verbose updates
            if self.verbose is not None:
                tqdm.write('Epoch %d, train loss = %.4f' % (e+1,total_loss.detach().item()))
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
                                                # extract input
                        images = x[0] # [B, T_img, C_img, W, H]
                        inerts = x[1] # [B, T_inert, C_inert]
                        yaw_hist = x[2]
                        # get values from dimensions
                        B = images.shape[0]
                        num_deltas = y.shape[2]
                        
                        # cast inputs to device
                        images = images.to(device=self.device, dtype=torch.float32)
                        inerts = inerts.to(device=self.device, dtype=torch.float32)
                        yaw_hist = yaw_hist.to(device=self.device, dtype=torch.float32)

                        # cast output to device
                        y = y.to(device=self.device, dtype=torch.float32) #[B, 3, num_deltas]

                        # initialize acummulated yaw
                        acc_yaw_pred = torch.zeros(B, 1, device=self.device)
                        acc_yaw_truth = torch.zeros(B, 1, device=self.device)
                        
                        # intialize other variables
                        trans_losses = []
                        rot_losses = []
                        for i in range (0,num_deltas):
                            # slice inputs to get indivual intervals 
                            image_input = images[:, i:i+2]
                            inert_input = inerts[:, i*4:i*4+5]
                            # yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                            # ==== SCHEDULED SAMPLING ====
                            if i == 0: # initialize with ground truth 
                                yaw_input = yaw_hist[:, i].unsqueeze(dim=1) + torch.randn_like(yaw_hist[:, i]) * 0.00876
                            # # elif random.random() < gt_probability: # if not the start of the interval, have a chance to still train from GT w/ noise (TODO: add noise)
                            # #     yaw_input = yaw_hist[:, i].unsqueeze(dim=1)
                            else: # else use the previous yaw estimate from model
                                yaw_input = yaw_est.detach().unsqueeze(dim=1)
                            # construct input to model
                            inputs = [image_input, inert_input, yaw_input]
                            # slice delta output target
                            target = y[:,:,i]
                            trans_target = target[:,0:2] # [B,2] first two predictions are translations
                            rot_target = target[:,2:3] #  [B,1] middle one is rotation
                            yaw_target = target[:,3:] # [B,2] last two are sin/cos comps of abs yaw
                            # call model
                            est = self.model(inputs)
                            # parse estimates
                            trans_est = est[0] # [B,2]
                            rot_est = est[1] # [B,1]
                            yaw_est = est[2] # [B,2]
                            # accumulate the yaw prediction over sequence length
                            acc_yaw_pred += rot_est
                            acc_yaw_truth += rot_target
                            # calculate delta losses
                            trans_loss = loss_func[0](trans_est, trans_target)
                            rot_loss = loss_func[1](rot_est, rot_target)
                            abs_yaw_loss = loss_func[2](yaw_est, yaw_target)
                            trans_losses.append(trans_loss.detach())
                            rot_losses.append(rot_loss.detach())
                            abs_yaw_losses.append(abs_yaw_loss.detach())
                            # Append estimates and truths if using for accuracy checking
                            if self.check_accuracy:
                                est_tensor =  torch.cat((trans_est, rot_est, yaw_est), dim=1)
                                val_est_history.append(est_tensor.detach())
                                val_truth_history.append(target.detach())
                        if self.check_accuracy:
                            val_acc_yaw_est_hist.append(acc_yaw_pred.detach())
                            val_acc_yaw_truth_hist.append(acc_yaw_truth.detach())
                        # take average of deltas over sequence interval
                        avg_trans_loss = torch.stack(trans_losses).mean()
                        avg_rot_loss = torch.stack(rot_losses).mean()
                        avg_yaw_loss = torch.stack(abs_yaw_losses).mean()
                        # compute accumalted yaw loss
                        acc_yaw_loss = loss_func[3](acc_yaw_pred, acc_yaw_truth)
                        # total_loss = self.loss_scale[0] * avg_trans_loss + self.loss_scale[1] * avg_rot_loss + self.loss_scale[2] * avg_yaw_loss
                        total_loss = self.loss_scale[0]*avg_trans_loss + self.loss_scale[1]*avg_rot_loss + self.loss_scale[2]*avg_yaw_loss + self.loss_scale[3]*acc_yaw_loss
                        
                        # log loss to history (if save interval not given then all losses are saved which can be expensive)
                        if self.loss_save_interval is None:
                            val_trans_loss_hist.append(avg_trans_loss.detach())
                            val_rot_lost_hist.append(avg_rot_loss.detach())
                            val_abs_yaw_loss_hist.append(avg_yaw_loss.detach())
                            val_acc_yaw_loss_hist.append(acc_yaw_loss.detach())
                            val_total_loss_hist.append(total_loss.detach())
                        elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
                            val_trans_loss_hist.append(avg_trans_loss.detach())
                            val_rot_lost_hist.append(avg_rot_loss.detach())
                            val_abs_yaw_loss_hist.append(avg_yaw_loss.detach())
                            val_acc_yaw_loss_hist.append(acc_yaw_loss.detach())
                            val_total_loss_hist.append(total_loss.detach())

                    # append loss history per epoch
                    val_epoch_loss_history.append(total_loss.detach())
            
                # verbose updates
                if self.verbose is not None:
                    tqdm.write('Epoch %d, val loss = %.4f' % (e+1,total_loss.detach().item()))
            #------------------------------ END OF VAL LOOP --------------------------
            
            
            #--------------- GET ACCURACIES, EARLY STOPPING, OVERFIT DETECTOR, AND SAVE WEIGHTS ---------------
            if self.check_accuracy:
                rmse_train_deltas = checkAccuracy(train_est_history, train_truth_history)
                # convert delta yaw pred to degrees (NOTE: Hacky)
                rmse_train_deltas[2] = np.rad2deg(rmse_train_deltas[2])
                rmse_train_acc_yaw = checkAccuracy(train_acc_yaw_est_hist, train_acc_yaw_truth_hist)
                rmse_train_acc_yaw = np.rad2deg(rmse_train_acc_yaw)
                rmse_train = np.hstack((rmse_train_deltas, rmse_train_acc_yaw))
                tqdm.write(f"Training RMSE: {rmse_train}")
                rmse_train_history.append(rmse_train)
                if self.run_val:
                    rmse_val_deltas = checkAccuracy(val_est_history, val_truth_history)
                    # convert delta yaw pred to degrees (NOTE: Hacky)
                    rmse_val_deltas[2] = np.rad2deg(rmse_val_deltas[2]) # change delta y from rad to degs
                    rmse_val_acc_yaw = checkAccuracy(val_acc_yaw_est_hist, val_acc_yaw_truth_hist)
                    rmse_val_acc_yaw = np.rad2deg(rmse_val_acc_yaw)
                    rmse_val = np.hstack((rmse_val_deltas, rmse_val_acc_yaw))
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
            del trans_loss, rot_loss, total_loss, acc_yaw_loss, est, trans_est, rot_est, acc_yaw_pred,
            if self.check_accuracy:
                del  rmse_train
                if self.run_val:
                    del rmse_val
            tqdm.write(f"Time per epoch: {time.time()-epoch_start_time}\n")
            # -------------------- END EPOCH ------------------------
        
        print("\nTraining Complete\n")

        # -------------------------- END TRAINING LOOP ---------------------------
        
        # convert loss list to cpu
        train_total_loss_hist = torch.stack(train_total_loss_hist).cpu().numpy()
        train_epoch_loss_history = torch.stack(train_epoch_loss_history).cpu().numpy() if epochs>1 else train_epoch_loss_history[0].item()
        train_trans_loss_hist = torch.stack(train_trans_loss_hist).cpu().numpy()
        train_rot_loss_hist = torch.stack(train_rot_loss_hist).cpu().numpy()
        train_abs_yaw_loss_hist = torch.stack(train_abs_yaw_loss_hist).cpu().numpy()
        train_acc_yaw_loss_hist = torch.stack(train_acc_yaw_loss_hist).cpu().numpy()
        
        if self.run_val:
            val_total_loss_hist = torch.stack(val_total_loss_hist).cpu().numpy()
            val_epoch_loss_history = torch.stack(val_epoch_loss_history).cpu().numpy() if epochs>1 else val_epoch_loss_history[0].item()
            val_trans_loss_hist = torch.stack(val_trans_loss_hist).cpu().numpy()
            val_rot_lost_hist = torch.stack(val_rot_lost_hist).cpu().numpy()
            val_abs_yaw_loss_hist = torch.stack(val_abs_yaw_loss_hist).cpu().numpy()
            val_acc_yaw_loss_hist = torch.stack(val_acc_yaw_loss_hist).cpu().numpy()
        
        if self.check_gradients:
            layer_idx, avg_grads = get_grads(grads)
            
        # package outs dict for returns
        outs = {"lr_history": lr_history,
                "train_total_loss_hist": train_total_loss_hist,
                "train_trans_loss_hist": train_trans_loss_hist,
                "train_rot_loss_hist": train_rot_loss_hist,
                "train_abs_yaw_loss_hist": train_abs_yaw_loss_hist,
                "train_acc_yaw_loss_hist": train_acc_yaw_loss_hist,
                "train_epoch_loss_history": train_epoch_loss_history,
            }
        if self.check_accuracy:
            outs["rmse_train_history"] = rmse_train_history
        if self.run_val:
            outs["val_total_loss_hist"] = val_total_loss_hist
            outs["val_trans_loss_hist"] = val_trans_loss_hist
            outs["val_rot_lost_hist"] = val_rot_lost_hist
            outs["val_abs_yaw_loss_hist"] = val_abs_yaw_loss_hist
            outs["val_acc_yaw_loss_hist"] = val_acc_yaw_loss_hist
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

def manual_mse_loss(est, truth, dim=0):
    '''
    A manual implementation of the the MSELoss to specify the
    dimension to take the mean along
    '''
    return ((est - truth)**2).mean(dim=dim)

def mse_yaw_angle_loss(pred, target):
    '''
    A custom MSE loss designed to handle yaw angle ambiguity
    '''
    diff = pred - target
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    
    return torch.mean(diff**2)

def get_gt_probability(epoch, total_epochs):
    '''
    Manually returns probability to train on GT based on the percentage of training
    '''
    if epoch < total_epochs * 0.3:
        return 0.9  # First 30%: mostly GT
    elif epoch < total_epochs * 0.6:
        # Middle 30%: gradual decay
        progress = (epoch - 0.3 * total_epochs) / (0.3 * total_epochs)
        return 0.9 - 0.6 * progress  # 0.9 → 0.3
    else:
        # Final 40%: low but non-zero GT
        return 0.2  # Still some GT to prevent collapse