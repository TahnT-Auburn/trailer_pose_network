import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class AsyncSpaceTimeYawHist(nn.Module):
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
        num_outputs = 3,
    ):
        super().__init__()
        assert img_size[0]*img_size[1] % patch_size == 0, \
            f"Input image size ({img_size[0],img_size[1]}) and patch size ({patch_size}) are not compatible."

        self.num_frames = num_frames
        self.num_imu_samples = num_imu_samples
        self.embed_dim = embed_dim
        self.num_patches = (img_size[0]*img_size[1]) // patch_size ** 2
        self.dropout = nn.Dropout(dropout)
        self.num_yaw_hist_samples = 1 # Assuming we omit the final (current) yaw
        
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
        
        # === YAW HISTORY EMBEDDING ===
        self.yaw_hist_embed = nn.Linear(2, embed_dim)
        
        # temporal positional embedding for yaw histories # UNDER CONSTRUCTION
        # self.yaw_hist_temporal_pos_embedding = nn.Parameter(
        #     torch.randn(1, self.num_yaw_hist_samples, embed_dim) * 0.02
        # )
        
        # === MODALITY TOKENS ===
        self.visual_mod_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.imu_mod_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        self.yaw_hist_mod_token = nn.Parameter(
            torch.randn(1, 1, embed_dim) * 0.02
        )
        
        # === SPACE-TIME TRANSFORMER BLOCKS FOR IMAGES ===
        self.visual_blocks = nn.ModuleList([
            SpaceTimeBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === TEMPORAL TRANSFORMER BLOCKS FOR IMU ===
        self.imu_blocks = nn.ModuleList([
            ImuTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        # === TEMPORAL TRANSFORMER BLOCKS FOR YAW HISTORIES ===
        self.yaw_hist_blocks = nn.ModuleList([
            YawHistTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === CROSS-MODAL ATTENTION
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === FUSION STRATEGY ===
        # self.fusion_strat = GlobalPoolFusionStrategy(embed_dim, dropout)
        self.fusion_strat = TaskSpecificFusionStrategy(embed_dim, num_heads, dropout)
        # self.fusion_strat = LightweightTaskFusion(embed_dim, dropout)
        
        # === NETWORK HEADS === 
        self.final_norm_trans = nn.LayerNorm(embed_dim)
        self.network_head_trans = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2), # output dx dy in body frame
        )

        self.final_norm_rot = nn.LayerNorm(embed_dim)
        self.network_head_rot = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1), # output dyaw
        )
        
        self.final_norm_yaw = nn.LayerNorm(embed_dim)
        self.network_head_yaw = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2), # sin cos yaw output 
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
    
    def embed_yaw_hist(self, yaw_histories):
        """Embeds yaw history input data.
        
        Args:
            yaw_histories (torch.tensor): [B, T, 1] Yaw histories.
        
        Returns:
            yaw_hist_tokens (torch.tensor): [B, T, D] Yaw history tokens.
        """
        B, T, _ = yaw_histories.shape
        
        # Linear projection to embedding dimension
        yaw_hist_tokens = self.yaw_hist_embed(yaw_histories)
        
        # Add temporal positional embedding
        # yaw_hist_tokens = yaw_hist_tokens + self.yaw_hist_temporal_pos_embedding
        
        # Add modality embedding
        yaw_hist_tokens = yaw_hist_tokens + self.yaw_hist_mod_token.expand(B, -1, -1)
        
        return yaw_hist_tokens
    
    def forward(self, x):
        """Forward block for model.

        Args:
            x (list): List of [B, T, C, H, W] raw images, [B, T, C] raw IMU data as tensors and [B, T, 1] yaw histories
        Returns:
            predictions (torch.tensor): model predictions
        """
        images = x[0]
        imu_data = x[1]
        yaw_histories = x[2]
        B = images.shape[0]
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)
        imu_tokens = self.embed_imu(imu_data)
        yaw_hist_tokens = self.embed_yaw_hist(yaw_histories)
        
        # Apply any dropout
        visual_tokens = self.dropout(visual_tokens)
        imu_tokens = self.dropout(imu_tokens)
        
        # === VISUAL SPACE-TIME ATTENTION ENCODER ===
        for block in self.visual_blocks:
            visual_tokens = block(visual_tokens, self.num_frames, self.num_patches)
            
        # === IMU TIME ATTENTION ENCODER ===
        for block in self.imu_blocks:
            imu_tokens = block(imu_tokens)
        
        # === YAW HIST TIME ATTENTION ENCODER === # UNDER CONSTRUCTION. TRYING A SINGLE HISTORY AND EMBEDDING ONLY
        # for block in self.yaw_hist_blocks:
        #     yaw_hist_tokens = block(yaw_hist_tokens) 
        
        # === CROSS-MODAL FUSION ===
        for cross_block in self.cross_modal_blocks:
            visual_tokens, imu_tokens, yaw_hist_tokens = cross_block(visual_tokens, imu_tokens, yaw_hist_tokens)
            
        # === FUSION OF VISUAL AND IMU TOKENS ===
        # fused_tokens = self.fusion_strat(visual_tokens, imu_tokens, yaw_hist_tokens)
        trans_feat, rot_feat, yaw_feat = self.fusion_strat(visual_tokens, imu_tokens, yaw_hist_tokens)
        
        # === FINAL NETWORK HEAD FOR PREDICTION ===
        trans_predictions = self.network_head_trans(self.final_norm_trans(trans_feat))
        rot_predictions = self.network_head_rot(self.final_norm_rot(rot_feat))
        yaw_predictions = self.network_head_yaw(self.final_norm_yaw(yaw_feat))
        # predictions =  torch.cat((trans_predictions, rot_predictions), dim=1)
        
        return trans_predictions, rot_predictions, yaw_predictions
    
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
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial_attn, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_spatial = x_spatial_attn + x_spatial # residual connection
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_spatial, '(b t) n d -> b (t n) d', b=B, t=num_frames)
        
        # TEMPORAL ATTENTION ===
        # Reshape to process each spatial location across time: [B*N_patches, T, D]
        x_temporal = rearrange(x, 'b (t n) d -> (b n) t d', t=num_frames, n=num_patches)
        
        # Self-attention across time for each spatial location
        x_temporal = self.temporal_norm(x_temporal)
        x_temporal_attn, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = x_temporal_attn + x_temporal # residual connection
        
        # Reshape back: [B, T*N_patches, D]
        x = rearrange(x_temporal, '(b n) t d -> b (t n) d', b=B, n=num_patches)
        
        # === FEED FORWARD ===
        x = self.mlp_norm(x)
        x_mlp = self.mlp(x)
        x = x_mlp + x # residual connection
        
        return x

class ImuTemporalBlock(nn.Module):
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
        x = self.norm1(x)
        x_out, _ = self.temporal_attn(x, x, x)
        x = x + x_out
        
        # MLP
        x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
        
        return x
    
class YawHistTemporalBlock(nn.Module):
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
        x: [B, T, D]
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
    
class CrossModalAttention(nn.Module):
    """Cross-attention between visual and IMU modalities"""
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        
        # Cross Attention
        self.visual_to_all_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.imu_to_all_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.yaw_hist_to_all_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feedforward
        self.visual_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.imu_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.yaw_hist_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.visual_norm1 = nn.LayerNorm(embed_dim)
        self.visual_norm2 = nn.LayerNorm(embed_dim)
        
        self.imu_norm1 = nn.LayerNorm(embed_dim)
        self.imu_norm2 = nn.LayerNorm(embed_dim)
        
        self.yaw_hist_norm1 = nn.LayerNorm(embed_dim)
        self.yaw_hist_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_tokens, imu_tokens, yaw_hist_tokens):
        """Forward block for cross attention

        Args:
            visual_tokens (torch.tensor): Tokens from images [B, T*N_patches, D]
            imu_tokens (torch.tensor): Tokens from IMU [B, T, D]
            yaw_hist_tokens (torch.tensor): Tokens from yaw history [B, T, D]
        Returns:
            fused_tokens (torch.tensor): Fused tokens via cross attention [B, combined_seq_len, D]
        """
        
        # === CROSS-MODAL ATTENTION ===
        # Concatenate modalities for key-value pairs
        imu_yaw_tokens = torch.cat([imu_tokens, yaw_hist_tokens], dim=1)
        visual_yaw_tokens = torch.cat([visual_tokens, yaw_hist_tokens], dim=1)
        visual_imu_tokens = torch.cat([visual_tokens, imu_tokens], dim=1)
        
        # Visual attending to IMU and Yaw tokens
        visual_tokens = self.visual_norm1(visual_tokens)
        visual_attended, _ = self.visual_to_all_attn(
            query=visual_tokens,
            key=imu_yaw_tokens,
            value=imu_yaw_tokens
        )
        visual_tokens = visual_tokens + visual_attended
        # Visual feedforward
        visual_tokens = self.visual_norm2(visual_tokens)
        visual_tokens_mlp = self.visual_mlp(visual_tokens)
        visual_tokens = visual_tokens + visual_tokens_mlp
        
        # IMU attending to visual and yaw
        imu_tokens = self.imu_norm1(imu_tokens)
        imu_attended, _ = self.imu_to_all_attn(
            query=imu_tokens,
            key=visual_yaw_tokens,
            value=visual_yaw_tokens
        )
        imu_tokens = imu_tokens + imu_attended
        # IMU feedforward
        imu_tokens = self.imu_norm2(imu_tokens)
        imu_tokens_mlp = self.imu_mlp(imu_tokens)
        imu_tokens = imu_tokens + imu_tokens_mlp
        
        # Yaw attending to visual and IMU
        yaw_hist_tokens = self.yaw_hist_norm1(yaw_hist_tokens)
        yaw_hist_attended, _ = self.yaw_hist_to_all_attn(
            query=yaw_hist_tokens,
            key=visual_imu_tokens,
            value=visual_imu_tokens
        )
        yaw_hist_tokens = yaw_hist_tokens + yaw_hist_attended
        # Yaw hist feedforward
        yaw_hist_tokens = self.yaw_hist_norm2(yaw_hist_tokens)
        yaw_hist_mlp = self.yaw_hist_mlp(yaw_hist_tokens)
        yaw_hist_tokens = yaw_hist_tokens + yaw_hist_mlp
        
        return visual_tokens, imu_tokens, yaw_hist_tokens
        
class GlobalPoolFusionStrategy(nn.Module):
    # === FUSION STRATEGY ===
    # Current strategy is to globally pool visual and imu tokens along the sequence dimension and concatenate along the embedding dimension
        def __init__(self, embed_dim, dropout):
            super().__init__()
            
            # Fusion Layer
            self.fusion_mlp = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            
        def forward(self, visual_tokens,  imu_tokens, yaw_hist_tokens):
            # Globally pool visual and imu tokens along the sequence dimension
            visual_pooled = visual_tokens.mean(dim=1)
            imu_pooled = imu_tokens.mean(dim=1)
            yaw_hist_pooled = yaw_hist_tokens.mean(dim=1)
            
            # Concatenate pooled tokens along the embedding dimesion
            combined_tokens = torch.cat([visual_pooled, imu_pooled, yaw_hist_pooled], dim=1)
            
            # Fuse combined tokens in MLP
            fused_tokens = self.fusion_mlp(combined_tokens) # [B, D]
            
            return fused_tokens
        
class TaskSpecificFusionStrategy(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        
        # learnable queries for each task
        # self.task_queries = nn.Parameter(torch.randn(3,1, embed_dim))
        self.translation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.rotation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.abs_yaw_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Cross-attention for pooling
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # small refinement mlps per task
        self.task_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        self.translation_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim) 
        )
        # small refinement mlps per task
        self.rotation_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        # small refinement mlps per task
        self.abs_yaw_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim)
        )

    def forward(self, visual_tokens,  imu_tokens, yaw_hist_tokens):
        B = visual_tokens.shape[0]
        
        # concatenate all tokens
        all_tokens = torch.cat([visual_tokens, imu_tokens, yaw_hist_tokens], dim=1)
        # generate features for each task
        # queries = self.task_queries.expand(-1, B, -1).transpose(0, 1) # [B, 3, D]
        
        # task_features, _ = self.attention(
        #     queries,
        #     all_tokens,
        #     all_tokens
        # )
        # task_features = self.task_mlp(task_features)
        # trans_feat = task_features[:,0]
        # rot_feat = task_features[:,1]
        # yaw_feat = task_features[:,2]
        # translation
        trans_feat, trans_attn = self.attention(
            self.translation_query.expand(B, -1, -1),
            all_tokens,
            all_tokens
        )
        trans_feat = trans_feat.squeeze(1)
        trans_feat = self.translation_mlp(trans_feat)
        # rotation
        rot_feat, rot_attn = self.attention(
            self.rotation_query.expand(B, -1, -1),
            all_tokens,
            all_tokens
        )
        rot_feat = rot_feat.squeeze(1)
        rot_feat = self.rotation_mlp(rot_feat)
        # abs yaw
        yaw_feat, yaw_attn = self.attention(
            self.abs_yaw_query.expand(B, -1, -1),
            all_tokens,
            all_tokens
        )
        yaw_feat = yaw_feat.squeeze(1)
        yaw_feat = self.abs_yaw_mlp(yaw_feat)
        
        return trans_feat, rot_feat, yaw_feat
    
    
class LightweightTaskFusion(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        
        # Task-specific attention weights (just linear layers)
        self.xy_attention = nn.Linear(embed_dim, 1)
        self.abs_yaw_attention = nn.Linear(embed_dim, 1)
        self.delta_yaw_attention = nn.Linear(embed_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def compute_task_feature(self, all_tokens, attention_layer):
        """
        Args:
            all_tokens: [B, N, D]
            attention_layer: Linear(D, 1)
        Returns:
            feature: [B, D]
        """
        # Compute attention weights
        attn_logits = attention_layer(all_tokens)  # [B, N, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B, N, 1]
        
        # Weighted sum
        feature = (all_tokens * attn_weights).sum(dim=1)  # [B, D]
        
        return feature, attn_weights
        
    def forward(self, visual_tokens, imu_tokens, yaw_tokens):
        """
        Args:
            visual_tokens: [B, N_v, D]
            imu_tokens: [B, N_i, D]
            yaw_tokens: [B, N_y, D]
        """
        # Concatenate once
        all_tokens = torch.cat([visual_tokens, imu_tokens, yaw_tokens], dim=1)
        
        # Compute task-specific features in parallel
        xy_feat, xy_attn = self.compute_task_feature(all_tokens, self.xy_attention)
        abs_yaw_feat, abs_yaw_attn = self.compute_task_feature(all_tokens, self.abs_yaw_attention)
        delta_yaw_feat, delta_yaw_attn = self.compute_task_feature(all_tokens, self.delta_yaw_attention)
        
        return xy_feat, delta_yaw_feat, abs_yaw_feat