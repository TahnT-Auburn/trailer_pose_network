'''
################  Transfer Learning using ViT Test ################
#
#               FOR TEST DRIVEN DEVELOPEMENT
#
###################################################################
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

from trailer_pose_network.data_setup import TrailerData
from trailer_pose_network.trainer import Trainer

TRAIN_CSV = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\reduced\\reduced_data_v2.csv'
OUT_PATH = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\weights\\tests\\vit_b_16_weights.pth'
TRAIN_SPECS = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\spreadsheets\\train_specs\\ablations\\network_head\\large\\vit_b_16_train_specs.csv'

#%%
##### load pretrained ViT #####

model = torchvision.models.vit_b_16(weights="IMAGENET1K_V1",
                                    # image_size=512,
                                    # patch_size=16,
                                    )
hidden_dim = model.hidden_dim
num_features = model.num_classes
model.heads = nn.Sequential(     # samll head
    nn.Linear(hidden_dim,1)
    # nn.Linear(1000, 1)
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device in Use: %s" % device)
torch.cuda.empty_cache()
model = model.to(device)

#%%
##### Setup data #####

full_set = TrailerData(csv_file=TRAIN_CSV,
                       transform=transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (0.31274095, 0.30793244, 0.30711132))
                                #  transforms.Normalize((0.32997385, 0.3125009, 0.3013311), (1, 1, 1)),
                                ]),
                                output_states=9)

# split training set to training/val sets
train_df = pd.read_csv(TRAIN_CSV)
# NUM_VAL = 1704
NUM_VAL = int(0.1*len(train_df)) # make validation set 10% of training set
NUM_TRAIN = int(len(full_set) - NUM_VAL)
train_set, val_set = random_split(full_set,[NUM_TRAIN, NUM_VAL])
# baby_set, train_set, val_set = random_split(full_set,[0.2,0.74,0.06])

BATCH_SIZE = 8
# generate loaders
loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
loader_val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
# loader_baby = DataLoader(baby_set, batch_size=BATCH_SIZE, shuffle=True)

#%%
##### visualize image patches #####

def extract_patches(image_tensor, patch_size=8):
    # get dimensions of the image tensor
    bs, c, h, w = image_tensor.size()

    # define the unfold layer with approproate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    # apply unfold to the image tensor
    unfolded = unfold(image_tensor)

    # reshape the unfolded tensor to match the desired output shape
    # Output shape: BSxLxH, where L is the number of patches in each dimension
    unfolded = unfolded.transpose(1,2).reshape(bs, -1, c*patch_size*patch_size)

    return unfolded

# view image patches
dataiter = next(iter(loader_train))  # create a dataloader itterable object
test_images, test_labels = dataiter # sample from the itterable object
patch_size = model.patch_size

# extract patches from the test images using the defined function
patches = extract_patches(test_images, patch_size=patch_size)
patches_square = patches.reshape(test_images.shape[0], -1, 3, patch_size, patch_size)

# calculate the grid size for visualization
grid_size = test_images.shape[2] // patch_size
print("Sequence Length %d" % grid_size**2)

# visulaize thhe patches as a grid
plt.figure(figsize=(5,5))
plt.axis('off')
out = torchvision.utils.make_grid(patches_square[0], grid_size, normalize=False, pad_value=0.5)
_ = plt.imshow(out.numpy().transpose((1,2,0)))
plt.tight_layout
plt.show()

#%% 
##### Setup Training #####

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

# define scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

print("Training model ...")
print()

network_trainer = Trainer(model=model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          loader_train=loader_train,
                          loader_val=loader_val,
                          loss_scale=1e2,
                          device=device,
                          verbose={"cond": True,"print_every":100},
                          save_weights={"cond": True, "save_path": OUT_PATH},
                          save_outs={"cond": True, "save_path": TRAIN_SPECS})

start_time = time.time()
outs = network_trainer.train(epochs=10)
print("Training Time: %s" % (time.time() - start_time))

# plot loss
plt.subplot(3,1,1)
plt.plot(outs["loss_history"], '-')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plot accuracies
plt.subplot(3,1,2)
plt.plot(outs["rmse_train_history"], '-o')
plt.plot(outs["rmse_val_history"], '-o')
plt.legend(['train', 'val'], loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('RMSE [deg]')
plt.tight_layout()
plt.show()

#%%
##### Visualize the positional embeddings #####

# extract positional embeddings from model
pos_embs = model.encoder.pos_embedding.detach().cpu()
sequence_length = model.seq_length
hidden_dim = model.hidden_dim
# caculate the cosine simularity between every positional embedding
dist = F.cosine_similarity(pos_embs[:,1:,:], pos_embs[:,1:,:].reshape(sequence_length-1,1,hidden_dim), dim=-1).numpy()
n_rows_cols = model.image_size//patch_size

# plot positional embeddings
fig, axes = plt.subplots(n_rows_cols, n_rows_cols, figsize=(5,5))
for i in range(n_rows_cols):
    for j in range(n_rows_cols):

        # generate a sample image
        img = dist[j + i * n_rows_cols].reshape(n_rows_cols, n_rows_cols)

        # display the image
        axes[i,j].imshow(img)
        axes[i,j].axis('off')

plt.tight_layout
plt.show()

# %%
