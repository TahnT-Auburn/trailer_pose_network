import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class AsyncSpaceTimeCrossAttention(nn.Module):
    def __init__(
        self,
        img_size = (224,224),
        patch_size = 16,
        in_channels = 3,
        embed_dim = 768,
        num_frames = 2,
        num_imu_samples = 5,
        imu_channels = 6,
        num_heads = 12,
        depth = 12, # TODO: make depth a list which spans all three transformers used
        dropout = 0.,
        num_outputs = 3
    ):
        super().__init__()
        assert img_size[0]*img_size[1] % patch_size == 0, \
            f"Input image size ({img_size[0],img_size[1]}) and patch size ({patch_size}) are not compatible."
        self.num_frames = num_frames
        self.num_imu_samples = num_imu_samples
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0]*img_size[1]) // patch_size ** 2
        self.dropout = nn.Dropout(dropout)
        
        # === IMAGE EMBEDDING ===
        # Patch embedding for spatial tokenization
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embeddings (spatial and temporal)
        self.spatial_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        self.temporal_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_frames, embed_dim) * 0.02
        )
        
        # === IMU EBEDDING ===
        # Linear projection for IMU features
        self.imu_embed = nn.Linear(imu_channels, embed_dim)
        
        # temporal positional embedding for IMU
        self.imu_temporal_pos_embedding = nn.Parameter(
            torch.randn(1, self.num_imu_samples, embed_dim) * 0.02
        )
        
        # === MODALITY TOKENS ===
        self.visual_mod_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.imu_mod_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        
        # === SPACE-TIME TRANSFORMER BLOCKS FOR IMAGES ===
        self.visual_blocks = nn.ModuleList([
            SpaceTimeBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === TEMPORAL TRANSFORMER BLOCKS FOR IMU ===
        self.imu_blocks = nn.ModuleList([
            TemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        # === CROSS-MODAL ATTENTION
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout=dropout)
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
        
        # Process all frames: [B*T, C, H, W] -> [B*T, D, H_p, W_p]
        images_flat = rearrange(images, 'b t c h w -> (b t) c h w')
        
        # Patch embedding: [B*T, D, H_p, W_p]
        patches = self.patch_embed(images_flat)

        # Flatten spatial patches: [B*T, N_patches, D]
        patches = rearrange(patches, 'bt d h w -> bt (h w) d')
        
        # Reshape back: [B, T, N_patches, D]
        patches = rearrange(patches, '(b t) n d -> b t n d', b=B, t=T)

        # Add spatial positional embedding
        patches = patches + self.spatial_pos_embedding.unsqueeze(1)
        
        # Flatten for transformer: [B, T*N_patches, D]
        visual_tokens = rearrange(patches, 'b t n d -> b (t n) d')
        
        # Add temporal positional embedding (repeat for each patch)
        temporal_pos = repeat(self.temporal_pos_embedding, 'b t d -> b (t n) d', n=self.num_patches)
        visual_tokens = visual_tokens + temporal_pos
        
        # Add modality tokens
        visual_tokens = visual_tokens + self.visual_mod_token.expand(B, -1, -1)
        
        return visual_tokens        
    
    def embed_imu(self, imu_data):
        """Embeds raw IMU input data.

        Args:
            imu_data (torch.tensor): [B, T, C] Raw imu data where C are the number of unique IMU sensor/axes

        Returns:
            imu_tokens (torch.tensor): [B, T, D] IMU tokens.
        """
        B, T, _ = imu_data.shape
        
        # Linear projection to embedding dimension
        imu_tokens = self.imu_embed(imu_data)
        
        # Add temporal positional embedding
        imu_tokens = imu_tokens + self.imu_temporal_pos_embedding
        
        # Add modality embedding
        imu_tokens = imu_tokens + self.imu_mod_token.expand(B, -1, -1)
        
        return imu_tokens
    
    def forward(self, x):
        """Forward block for model.

        Args:
            x (list): List of [B, T, C, H, W] raw images and [B, T, C] raw IMU data as tensors
        Returns:
            predictions (torch.tensor): model predictions
        """
        images = x[0]
        imu_data = x[1]
        B = images.shape[0]
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)
        imu_tokens = self.embed_imu(imu_data)
        
        # Apply any dropout
        visual_tokens = self.dropout(visual_tokens)
        imu_tokens = self.dropout(imu_tokens)
        
        # === VISUAL SPACE-TIME ATTENTION ENCODER ===
        for block in self.visual_blocks:
            visual_tokens = block(visual_tokens, self.num_frames, self.num_patches)
            
        # === IMU TIME ATTENTION ENCODER ===
        for block in self.imu_blocks:
            imu_tokens = block(imu_tokens)
        
        # === CROSS-MODAL FUSION ===
        for cross_block in self.cross_modal_blocks:
            visual_tokens, imu_tokens = cross_block(visual_tokens, imu_tokens)
            
        # === FUSION OF VISUAL AND IMU TOKENS ===
        fused_tokens = self.fusion_strat(visual_tokens, imu_tokens)
        
        # === FINAL NETWORK HEAD FOR PREDICTION ===
        trans_predictions = self.network_head_trans(self.final_norm_trans(fused_tokens))
        rot_predictions = self.network_head_rot(self.final_norm_rot(fused_tokens))
        # predictions =  torch.cat((trans_predictions, rot_predictions), dim=1)
        
        return trans_predictions, rot_predictions
    
class SpaceTimeBlock(nn.Module):
    """Factorized space-time attention block"""
    def __init__(self,
                 embed_dim, num_heads, dropout=0.):
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
        
        # Feed forward layer
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.mlp_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, num_frames, num_patches):
        """Space-time forward block

        Args:
            x (torch.tensor): [B, T*N_patches, D]
            num_frames (int): Number of image frames.
            num_patches (int): Number of image patches
        """
        B, seq_len, D = x.shape
        
        # === SPATIAL ATTENTION ===
        # Reshape the process each frame separately: [B*T, N_patches, D]
        x_spatial = rearrange(x, 'b (t n) d -> (b t) n d', t=num_frames, n=num_patches)
        
        # Self-attention within each frame
        # x_spatial = self.spatial_norm(x_spatial)
        x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial_attn + x_spatial # residual connection
        x_spatial = self.spatial_norm(x_spatial)
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_spatial, '(b t) n d -> b (t n) d', b=B, t=num_frames)
        
        # TEMPORAL ATTENTION ===
        # Reshape to process each spatial location across time: [B*N_patches, T, D]
        x_temporal = rearrange(x, 'b (t n) d -> (b n) t d', t=num_frames, n=num_patches)
        
        # Self-attention across time for each spatial location
        # x_temporal = self.temporal_norm(x_temporal)
        x_temporal_attn, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = x_temporal_attn + x_temporal # residual connection
        x_temporal = self.temporal_norm(x_temporal)
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_temporal, '(b n) t d -> b (t n) d', b=B, n=num_patches)
        
        # === FEED FORWARD ===
        # x = self.mlp_norm(x)
        x_mlp = self.mlp(x)
        x = x_mlp + x # residual connection
        x = self.mlp_norm(x)
        
        return x

class TemporalBlock(nn.Module):
    """Temporal attention for IMU tokens"""
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
        x: [B, T_imu, D]
        """
        # Temporal self-attention
        # x = self.norm1(x)
        x_out, _ = self.temporal_attn(x, x, x)
        x = x + x_out
        x = self.norm1(x)
        
        # MLP
        # x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
        x = self.norm2(x)
        
        return x
    
class CrossModalAttention(nn.Module):
    """Cross-attention between visual and IMU modalities"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        
        self.visual_to_imu_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.imu_to_visual_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.imu_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_tokens, imu_tokens):
        """Forward block for cross attention

        Args:
            visual_tokens (torch.tensor): Tokens from images [B, T*N_patches, D]
            imu_tokens (torch.tensor): Tokens from IMU [B, T, D]
        Returns:
            fused_tokens (torch.tensor): Fused tokens via cross attention [B, combined_seq_len, D]
        """
        
        # === CROSS-MODAL ATTENTION ===
        
        # Visual attending to IMU (what IMU info is relevant for visual features)
        visual_attended, _ = self.visual_to_imu_attn(
            query=visual_tokens,
            key=imu_tokens,
            value=imu_tokens
        )
        visual_tokens = visual_tokens + visual_attended
        visual_tokens = self.visual_norm(visual_tokens)
        
        # IMU attending to visual (what visual info is relevant for IMU features)
        imu_attended, _ = self.imu_to_visual_attn(
            query=imu_tokens,
            key=visual_tokens,
            value=visual_tokens
        )
        imu_tokens = imu_tokens + imu_attended
        imu_tokens = self.imu_norm(imu_tokens)
        
        return visual_tokens, imu_tokens
        
class FusionStrategy(nn.Module):
    # === FUSION STRATEGY ===
    # Current strategy is to globally pool visual and imu tokens along the sequence dimension and concatenate along the embedding dimension
        def __init__(self, embed_dim, dropout):
            super().__init__()
            
            # Fusion Layer
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            
        def forward(self, visual_tokens,  imu_tokens):
            # Globally pool visual and imu tokens along the sequence dimension
            visual_pooled = visual_tokens.mean(dim=1)
            imu_pooled = imu_tokens.mean(dim=1)
            
            # Concatenate pooled tokens along the embedding dimesion
            combined_tokens = torch.cat([visual_pooled, imu_pooled], dim=1)
            
            # Fuse combined tokens in MLP
            fused_tokens = self.fusion_mlp(combined_tokens) # [B, D]
            
            return fused_tokens