#%%
import torch

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import keyboard

class Trainer():
    def __init__(self,
    model,
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
        
    def train(self, loss_func, epochs=1):
        """
        Train function call
        """
        lr_history = []
        train_loss_history = []
        train_epoch_loss_history = []
        rmse_train_history = []
        if self.run_val:
            val_loss_history = []
            val_epoch_loss_history = []
            rmse_val_history = []
            
        self.model = self.model.to(device=self.device)
        
        # TODO: Add gradient checking here later. Must import functiosn accordingly
        
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

                y = y.to(device=self.device, dtype=torch.float32)

                # zero out all gradients for the variables which the optimizer will update
                self.optimizer.zero_grad()
                
                # call model to estimate
                est = self.model(x)
                
                # get learning rate
                lr = getLR(optimizer=self.optimizer)
                lr_history.append(lr)

                loss = loss_func(est, y)

                loss = self.loss_scale * loss

                # perform backward pass
                loss.backward()
                
                # Clip gradient
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # update the model using the computed gradients
                self.optimizer.step()
                
                # log loss to history (if save interval not given then all losses are saved which can be expensive)
                if self.loss_save_interval is None:
                    train_loss_history.append(loss.detach())
                elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
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
                        
                        y = y.to(device=self.device, dtype=torch.float32)

                        # call model to estimate
                        est = self.model(x)

                        loss = loss_func(est, y)
                        loss = self.loss_scale * loss

                        # log loss to history (if save interval not given then all losses are saved which can be expensive)
                        if self.loss_save_interval is None:
                            val_loss_history.append(loss.detach())
                        elif self.loss_save_interval is not None and t % self.loss_save_interval == 0:
                            val_loss_history.append(loss.detach())

                        # check the training and validation accuracies at the end of every epoch
                        if self.check_accuracy:
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
            
            tqdm.write(f"Time per epoch: {time.time()-epoch_start_time}\n")
            # -------------------- END EPOCH ------------------------
        
        print("\nTraining Complete\n")

        # convert loss list to cpu
        train_loss_history = torch.stack(train_loss_history).cpu().numpy()
        train_epoch_loss_history = torch.stack(train_epoch_loss_history).cpu().numpy() if epochs>1 else train_epoch_loss_history[0].item()
        if self.run_val:
            val_loss_history = torch.stack(val_loss_history).cpu().numpy()
            val_epoch_loss_history = torch.stack(val_epoch_loss_history).cpu().numpy() if epochs>1 else val_epoch_loss_history[0].item()
        
        # package outs dict for returns
        outs = {"lr_history": lr_history,
                "train_loss_history": train_loss_history,
                "train_epoch_loss_history": train_epoch_loss_history,
            }
        if self.check_accuracy:
            outs["rmse_train_history"] = rmse_train_history
        if self.run_val:
            outs["val_loss_history"] = val_loss_history
            outs["val_epoch_loss_history"] = val_epoch_loss_history
            if self.check_accuracy:
                outs["rmse_val_history"] = rmse_val_history

        if self.save_outs is not None:    
            # df = pd.DataFrame(outs)
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in outs.items()]))
            df.to_csv(self.save_outs["save_path"])
            print("Training outs saved to: %s" % self.save_outs)

        return self.model, outs    
            
            
            
            
            

############# UTILITY FUNCTIONS #############

def getLR(optimizer):
    '''
    Gets current all learning rate values from optimizer.
    '''
    # for param
    return optimizer.param_groups[0]['lr']

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
    # rmse = np.rad2deg(rmse)

    return rmse
    
def get_rmse(x_true, x_pred, dim=0):
    '''
    Calculates root mean squared error (RMSE)
    '''
    return torch.sqrt(torch.mean((x_true - x_pred)**2, dim=dim))

