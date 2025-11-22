import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNetBasedVisualOdom(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        
    ):
        super().__init__()
        resnet_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        # prepare resnet model with custom head
        in_feats = resnet_model.fc.in_features
        # replace resnet classification head with custom head. 
        resnet_model.fc = nn.Sequential(
            nn.Linear(in_feats, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )
        self.resnet_encoder = resnet_model
        
        # feature cross attention
        self.feat_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # fusion head       
        # Expects concatenated features from both images
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
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
        feat1 = self.resnet_encoder(img1) #[B, D]
        feat2 = self.resnet_encoder(img2) #[B, D]
        
        # unsqueeze feature for sequence dim of 1 from cross attention
        feat1_seq = feat1.unsqueeze(1)
        feat2_seq = feat2.unsqueeze(1)
        
        feat1_attended, _ = self.feat_cross_attn(
            feat1_seq,
            feat2_seq,
            feat2_seq
        )
        feat1_attended = feat1_attended.squeeze(1)
        
        feat2_attended, _ = self.feat_cross_attn(
            feat2_seq,
            feat1_seq,
            feat1_seq
        )
        feat2_attended = feat2_attended.squeeze(1)

        # Concatenate features
        combined_feats = torch.cat([feat1_attended, feat2_attended], dim=1) #[B, D*2]
        
        # Pass to fusion head and predict rotation
        delta_yaw_sincos = self.fusion_head(combined_feats)
        
        return delta_yaw_sincos
        