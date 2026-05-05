import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNetBasedSelfAttention(nn.Module):
    def __init__(
        self,
        resnet_model,
        num_frames,
        embed_dim,
        num_heads,
        depth,
        dropout,
    ):
        super().__init__()
        in_feats = resnet_model.fc.in_features
        # replace resnet classification head with custom head. 
        resnet_model.fc = nn.Sequential(
            nn.Linear(in_feats, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.resnet_encoder = resnet_model
        self.num_frames = num_frames
        self.dropout = nn.Dropout(dropout)
        # === TRANSFORMER BLOCKS FOR IMAGES ===
        self.visual_blocks = nn.ModuleList([
            VisualTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === FUSION STRATEGY ===
        self.fusion_strat = FusionStrategy(embed_dim, dropout)
        
        # === NETWORK HEADS === 
        self.final_norm_trans = nn.LayerNorm(embed_dim)
        self.network_head_trans = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        self.final_norm_rot = nn.LayerNorm(embed_dim)
        self.network_head_rot = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        
    def embed_images(self, images):
        """Embeds raw image inputs.

        Args:
            images (torch.tensor): [B, T, C, H, W] Raw images

        Returns:
            visual_tokens (torch.tensor): [B, T*N_patches, D] Visual tokens
        """
        B, T, C, H, W = images.shape
        
        # generate visual features for each image and stack for full visual token
        feats = []
        for i in range(0,self.num_frames):
            img = images[:,i]
            feat = self.resnet_encoder(img)
            feats.append(feat)
        visual_tokens = torch.stack(feats, dim=1) #[B, T, D]
        
        return visual_tokens
    
    def forward(self, x):
        images = x[0]
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)
        # Apply any dropout
        visual_tokens = self.dropout(visual_tokens)

        # === VISUAL FEATURE TIME ATTENTION ENCODER ===
        for block in self.visual_blocks:
            visual_tokens = block(visual_tokens)
        
        # === FUSION OF VISUAL AND IMU TOKENS ===
        fused_tokens = self.fusion_strat(visual_tokens)
        
        # trans_predictions = self.network_head_trans(self.final_norm_trans(fused_tokens))
        rot_predictions = self.network_head_rot(self.final_norm_rot(fused_tokens))
        
        return rot_predictions
    
class VisualTemporalBlock(nn.Module):
    """Temporal attention for visual tokens"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: [B, T_visual, D]
        """
        # Temporal self-attention
        x = self.norm1(x)
        x_out, _ = self.temporal_attn(x, x, x)
        x = x + x_out
        
        # MLP
        x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
        
        return x
    
class FusionStrategy(nn.Module):
    # === FUSION STRATEGY ===
    # Current strategy is to globally pool visual and imu tokens along the sequence dimension and concatenate along the embedding dimension
        def __init__(self, embed_dim, dropout):
            super().__init__()
            
            # Fusion Layer
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim , embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            
        def forward(self, visual_tokens):
            # Globally pool visual and imu tokens along the sequence dimension
            visual_pooled = visual_tokens.mean(dim=1)
            
            # Fuse combined tokens in MLP
            fused_tokens = self.fusion_mlp(visual_pooled) # [B, D]
            
            return fused_tokens