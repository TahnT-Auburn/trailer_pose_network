import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import flow_to_image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchvision.models.optical_flow import raft_large

from trailer_pose_network.dataloaders.asynchronous_temporal_dataloader import AsyncTemporalDataLoader

SEQ_ROOT_PROCESSED = "D:\\TrainingData\\experimental\\10Hz\\original\\"
SEQ_ROOT_RAW = "D:\\TrainingData\\experimental\\40Hz\\original\\"

# === DATALOADER PARAMETERS ===
NUM_FRAMES = 2
IMG_SIZE = (224,448)
BATCH_SIZE = 2
NUM_WORKERS = 0

def test():
    test_set = AsyncTemporalDataLoader(
        sequence_root_processed=SEQ_ROOT_PROCESSED,
        sequence_root_raw=SEQ_ROOT_RAW,
        sequential_lookback=NUM_FRAMES,
        inputs={'cam':True, 'can':False, 'imu':False, 'yaw_hist':False},
        reduce={'target_column':'steer_ang', 'target_size':50},
        transform_img=T.Compose([
            T.ToPILImage(),
            T.Resize(IMG_SIZE),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]),
    )
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # load raft
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    
    with torch.no_grad():
        for t, (x,y) in enumerate(tqdm(test_loader)):
            x = x[0].to(device=device, dtype=torch.float32)
            img_batch1 = x[:,0].to(device)
            img_batch2 = x[:,1].to(device)
            # plot(img_batch1)
            
            list_of_flows = model(img_batch1, img_batch2)
            predicted_flows = list_of_flows[-1] # takes the last of the flow prediction (apparently the best one)
            
            # visualize flow
            flow_imgs = flow_to_image(predicted_flows)
            # remap to [0 1]
            img_batch1 = [(img1 + 1) / 2 for img1 in img_batch1]
            grid = [[img1, flow_img] for (img1, flow_img) in zip(img_batch1, flow_imgs)]
            plot(grid)
            stop = 1        
def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test()