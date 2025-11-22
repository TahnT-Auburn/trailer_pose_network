import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.optical_flow import raft_large
class RaftBasedVisualOdom(nn.Module):
    def __init__(
        self,
        embed_dim,
    ):
        super().__init__()
        self.raft_model = raft_large(pretrained=True, progress=False)
        # Freeze raft
        for param in self.raft_model.parameters():
            param.requires_grad = False

        # adaptive pooling
        self.flow_pooler = nn.AdaptiveAvgPool2d((8,8))
        
        # network head to take flow estimates       
        self.fusion_head = nn.Sequential(
            nn.Linear(2 * 8 * 8, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2) # returns sin/cos delta yaw prediction
        )

    def forward(self, x):
        """
        Args:
            image_pair (torch.tensor): [B, 2, C, H, W] two consecutive frames
            
        Returns:
            delta_yaw_sincos: [B, 2] sin/cos component of delta yaw rotation.
        """
        image_pair = x[0]
        B = image_pair.shape[0]
        
        # Extract features from both images using pretrained resnet
        img1 = image_pair[:, 0]
        img2 = image_pair[:, 1]

        # Get flow
        with torch.no_grad():
            flow_list = self.raft_model(img1, img2)
            flow = flow_list[-1] # [B, 2, H, W]

        # downsample
        flow_downsampled = self.flow_pooler(flow) # [B, 2, 8, 8]
        
        # flatten spatial dimensions
        flow_flat = flow_downsampled.view(flow_downsampled.size(0), -1) # [B, 128]
        # Pass to fusion head and predict rotation
        delta_yaw_sincos = self.fusion_head(flow_flat)
        
        return delta_yaw_sincos
        