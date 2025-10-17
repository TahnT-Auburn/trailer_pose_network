'''
Space Time Attention Transformer that fuses Camera and IMU via cross attention.

Author: Tahn Thawainin
        Email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class SpaceTimeCrossAttention(nn.Module):
    def __init__(
        self,
        num_outputs,
        img_size=(224,224),
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        num_frames=15,  # Same for both modalities since they're synced
        imu_channels=8,  # 3-axis gyro + 3-axis accel
        num_heads=12,
        depth=12,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0]*img_size[1]) // patch_size ** 2
        
        # === IMAGE EMBEDDING ===
        # Patch embedding for spatial tokenization
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embeddings
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_frames, embed_dim) * 0.02
        )
        
        # === IMU EMBEDDING ===
        # Linear projection for IMU features
        self.imu_embed = nn.Linear(imu_channels, embed_dim)
        
        # Temporal positional embedding for IMU (same as visual since synced)
        self.imu_temporal_pos = nn.Parameter(
            torch.randn(1, num_frames, embed_dim) * 0.02
        )
        
        # === MODALITY TOKENS ===
        self.visual_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.imu_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # === SPACE-TIME TRANSFORMER BLOCKS ===
        self.space_time_blocks = nn.ModuleList([
            SpaceTimeBlock(embed_dim, num_heads, dropout) 
            for _ in range(depth // 2)
        ])
        
        # === CROSS-MODAL FUSION ===
        self.cross_attention = CrossModalAttention(embed_dim, num_heads, dropout)
        
        # === OUTPUT HEAD ===
        self.norm = nn.LayerNorm(embed_dim)
        self.yaw_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_outputs)
        )  # Predict single yaw angle
        
        self.dropout = nn.Dropout(dropout)
        
    def embed_images(self, images):
        """
        Embed image sequence into space-time tokens
        images: [B, T, C, H, W]
        returns: [B, T*N_patches, D]
        """
        B, T, C, H, W = images.shape
        
        # Reshape for batch processing: [B*T, C, H, W]
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
        
        # Patch embedding: [B*T, D, H_p, W_p]
        patches = self.patch_embed(images_flat)
        
        # Flatten spatial patches: [B*T, N_patches, D]
        patches = rearrange(patches, 'bt d h w -> bt (h w) d')
        
        # Reshape back to sequence: [B, T, N_patches, D]
        patches = rearrange(patches, '(b t) n d -> b t n d', b=B, t=T)
        
        # Add spatial positional embedding
        patches = patches + self.spatial_pos_embed.unsqueeze(1)  # [B, T, N_patches, D]
        
        # Flatten for transformer: [B, T*N_patches, D]
        visual_tokens = rearrange(patches, 'b t n d -> b (t n) d')
        
        # Add temporal positional embedding (repeat for each patch)
        temporal_pos = repeat(self.temporal_pos_embed, 'b t d -> b (t n) d', n=self.num_patches)
        visual_tokens = visual_tokens + temporal_pos
        
        return visual_tokens
    
    def embed_imu(self, imu_data):
        """
        Embed synchronized IMU sequence into temporal tokens
        imu_data: [B, T, 6] - SAME T as image frames since they're synced
        returns: [B, T, D]
        """
        B, T, _ = imu_data.shape
        
        # Linear projection: [B, T, D]
        imu_tokens = self.imu_embed(imu_data)
        
        # Add temporal positional embedding (same temporal grid as visual)
        imu_tokens = imu_tokens + self.imu_temporal_pos
        
        return imu_tokens
    
    def forward(self, inputs):
        """
        images: [B, T, C, H, W] - e.g., [B, 15, 3, 224, 224]
        imu_data: [B, T, 6] - e.g., [B, 15, 6] - SAME T since synced!
        returns: yaw_angle [B, 1]
        """
        images = inputs[0]
        imu_data = inputs[1]
        
        B = images.shape[0]
        
        # Verify sync - both should have same temporal dimension
        assert images.shape[1] == imu_data.shape[1], f"Synced data should have same T: images {images.shape[1]} vs IMU {imu_data.shape[1]}"
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)  # [B, T*N_patches, D]
        imu_tokens = self.embed_imu(imu_data)      # [B, T, D] - Same T!
        
        # Add modality tokens
        visual_tokens = visual_tokens + self.visual_token.expand(B, -1, -1)
        imu_tokens = imu_tokens + self.imu_token.expand(B, -1, -1)
        
        # Apply dropout
        visual_tokens = self.dropout(visual_tokens)
        imu_tokens = self.dropout(imu_tokens)
        
        # === SPACE-TIME PROCESSING ===
        # Process visual tokens through space-time blocks
        for block in self.space_time_blocks:
            visual_tokens = block(visual_tokens, self.num_frames, self.num_patches)
        
        # === CROSS-MODAL FUSION ===
        # Fuse visual and IMU information (perfectly aligned temporal grids)
        fused_tokens = self.cross_attention(visual_tokens, imu_tokens)  # [B, seq_len, D]
        
        # === PREDICTION HEAD ===
        # Global average pooling over sequence dimension
        global_repr = fused_tokens.mean(dim=1)  # [B, D]
        
        # Layer norm and prediction
        global_repr = self.norm(global_repr)
        yaw_angle = self.yaw_head(global_repr)  # [B, 1]
        
        return yaw_angle


class SpaceTimeBlock(nn.Module):
    """Factorized space-time attention block"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Spatial attention (within each frame)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(embed_dim)
        
        # Temporal attention (across frames for each spatial location)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        
        # Feed forward
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, num_frames, num_patches):
        """
        x: [B, T*N_patches, D]
        """
        B, seq_len, D = x.shape
        
        # === SPATIAL ATTENTION ===
        # Reshape to process each frame separately: [B*T, N_patches, D]
        x_spatial = rearrange(x, 'b (t n) d -> (b t) n d', t=num_frames, n=num_patches)
        
        # Self-attention within each frame
        x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial + x_spatial_attn
        x_spatial = self.spatial_norm(x_spatial)
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_spatial, '(b t) n d -> b (t n) d', b=B, t=num_frames)
        
        # === TEMPORAL ATTENTION ===
        # Reshape to process each spatial location across time: [B*N_patches, T, D]
        x_temporal = rearrange(x, 'b (t n) d -> (b n) t d', t=num_frames, n=num_patches)
        
        # Self-attention across time for each spatial location
        x_temporal_attn, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = x_temporal + x_temporal_attn
        x_temporal = self.temporal_norm(x_temporal)
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_temporal, '(b n) t d -> b (t n) d', b=B, n=num_patches)
        
        # === FEED FORWARD ===
        x_mlp = self.mlp(x)
        x = x + x_mlp
        x = self.mlp_norm(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-attention between visual and IMU modalities"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.visual_to_imu_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.imu_to_visual_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.imu_norm = nn.LayerNorm(embed_dim)
        
        # Fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, visual_tokens, imu_tokens):
        """
        visual_tokens: [B, T*N_patches, D]
        imu_tokens: [B, T, D]
        returns: [B, combined_seq_len, D]
        """
        
        # === CROSS-MODAL ATTENTION ===
        
        # Visual attending to IMU (what IMU info is relevant for visual features?)
        visual_attended, _ = self.visual_to_imu_attn(
            query=visual_tokens,
            key=imu_tokens,
            value=imu_tokens
        )
        visual_enhanced = visual_tokens + visual_attended
        visual_enhanced = self.visual_norm(visual_enhanced)
        
        # IMU attending to Visual (what visual info is relevant for IMU features?)
        imu_attended, _ = self.imu_to_visual_attn(
            query=imu_tokens,
            key=visual_tokens, 
            value=visual_tokens
        )
        imu_enhanced = imu_tokens + imu_attended
        imu_enhanced = self.imu_norm(imu_enhanced)
        
        # === FUSION STRATEGY ===
        
        # Since data is synced, temporal alignment is perfect!
        B, visual_len, D = visual_enhanced.shape  # [B, T*N_patches, D]
        B, imu_len, D = imu_enhanced.shape        # [B, T, D] - Same T as visual frames
        
        # Expand IMU to match visual spatial-temporal resolution
        # Each IMU sample corresponds to one frame, expand to all patches in that frame
        patches_per_frame = visual_len // imu_len  # Should equal N_patches
        imu_expanded = imu_enhanced.unsqueeze(2).expand(-1, -1, patches_per_frame, -1)  # [B, T, N_patches, D]
        imu_expanded = rearrange(imu_expanded, 'b t n d -> b (t n) d')  # [B, T*N_patches, D]
        
        # Now both have same sequence length: [B, T*N_patches, D]
        assert visual_enhanced.shape == imu_expanded.shape, "Synced data should align perfectly"
        
        # Concatenate features
        combined_features = torch.cat([visual_enhanced, imu_expanded], dim=-1)  # [B, T*N_patches, 2*D]
        
        # Fuse with MLP
        fused_tokens = self.fusion_mlp(combined_features)  # [B, T*N_patches, D]
        
        return fused_tokens


# # === USAGE EXAMPLE ===
# if __name__ == "__main__":
#     # Model parameters for synced data
#     model = VisualIMUYawTransformer(
#         img_size=224,
#         patch_size=16,
#         embed_dim=768,
#         num_frames=15,  # Same for both modalities since synced
#         num_heads=12,
#         depth=12
#     )
    
#     # Example input - SYNCED data (same temporal dimension)
#     batch_size = 4
#     images = torch.randn(batch_size, 15, 3, 224, 224)  # [B, T, C, H, W]
#     imu_data = torch.randn(batch_size, 15, 6)          # [B, T, 6] - Same T!
    
#     # Forward pass
#     yaw_prediction = model(images, imu_data)  # [B, 1]
#     print(f"Input shapes: images {images.shape}, IMU {imu_data.shape}")
#     print(f"Visual tokens: {15 * (224//16)**2} = {15 * 196} tokens")
#     print(f"IMU tokens: {15} tokens (perfectly synced!)")
#     print(f"Output shape: {yaw_prediction.shape}")
#     print(f"Predicted yaw angles: {yaw_prediction.squeeze()}")
    