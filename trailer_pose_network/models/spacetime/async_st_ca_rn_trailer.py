import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import torchvision
import copy

class AsyncSpaceTimeCrossAttentionResNet(nn.Module):
    def __init__(
        self,
        resnet_model:nn.Module,
        resnet_model_hitch:nn.Module,
        img_size:tuple = (224,224),
        seqential_lookback:int = 2,
        num_deltas:int = 1,
        embed_dim:int = 768,
        num_frames:int = 2,
        num_imu_samples:int = 5,
        imu_channels:int = 6,
        num_heads:int = 12,
        depth:int = 12,
        dropout:float = 0.,
        modality_dropout: dict | None = None,
    ): 
        super().__init__()
        assert img_size[0]*img_size[1] % 8 == 0, \
            f"Input image size ({img_size[0],img_size[1]}) are not compatible with ResNet encoder. W,H must be divisible my 8."
        self.sequential_lookback = seqential_lookback
        self.num_deltas = num_deltas
        self.num_frames = num_frames
        self.num_imu_samples = num_imu_samples
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.modality_dropout = modality_dropout
        
        # === IMAGE RESNET ENCODER FOR HITCH PREDICTION ====
        re_h_in_feats = resnet_model_hitch.fc.in_features # classifier for mobilenet
        resnet_model_hitch.fc = nn.Sequential(
            nn.Linear(re_h_in_feats, embed_dim),
            nn.GELU(),
        )
        self.hitch_resnet_encoder = resnet_model_hitch
        
        # === IMAGE RESNET ENCODER ===
        # prepare resnet model with custom head
        in_feats = resnet_model.fc.in_features
        # replace resnet classification head with custom head. 
        resnet_model.fc = nn.Sequential(
            nn.Linear(in_feats, embed_dim),
            nn.GELU(),
            # nn.Linear(embed_dim, embed_dim * 2),
            # nn.GELU(),
            # nn.Linear(embed_dim * 2, embed_dim),
            # nn.GELU(),
        )
        self.resnet_encoder = resnet_model
        
        self.visual_temporal_pos_embedding = nn.Parameter(
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
        
        # === TEMPORAL TRANSFORMER BLOCKS FOR IMAGE FEATURES ===
        self.visual_blocks = nn.ModuleList([
            VisualTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === TEMPORAL TRANSFORMER BLOCKS FOR IMU FEATURES ===
        self.imu_blocks = nn.ModuleList([
            ImuTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])

        # === CROSS-MODAL ATTENTION
        self.cross_modal_blocks = nn.ModuleList([
            CrossModalAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === FUSION STRATEGY ===
        # self.fusion_strat = TaskSpecificFusionStrategy(embed_dim, num_heads, dropout)
        self.fusion_strat = GloabalPoolFusionStrategy(embed_dim, dropout)
        
        # === NETWORK HEADS === 
        self.final_norm_trans = nn.LayerNorm(embed_dim)
        self.network_head_trans = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * self.num_deltas), # translation
        )

        self.final_norm_rot = nn.LayerNorm(embed_dim)
        self.network_head_rot = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1 * self.num_deltas), # rotation
        )
        
        self.final_norm_hitch = nn.LayerNorm(embed_dim)
        self.network_head_hitch = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1) # hitch angle
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
        visual_tokens = torch.stack(feats, dim=1)

        # add temporal positional embedding
        visual_tokens = visual_tokens + self.visual_temporal_pos_embedding

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
        latest_image = images[:,-1]
        
        # === HITCH ENCODER ===
        hitch_tokens = self.hitch_resnet_encoder(latest_image)
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)
        imu_tokens = self.embed_imu(imu_data)
        
        # Apply any dropout
        visual_tokens = self.dropout(visual_tokens)
        imu_tokens = self.dropout(imu_tokens)
        
        # Apply modality dropout
        if self.training:
            if self.modality_dropout is not None:
                if torch.rand(1).item() < self.modality_dropout["imu_dropout_rate"]:
                    # print("IMU Modality Dropout!")
                    imu_tokens = torch.zeros_like(imu_tokens)
                if torch.rand(1).item() < self.modality_dropout["cam_dropout_rate"]:
                    visual_tokens = torch.zeros_like(visual_tokens)
                    # print("Camera Modality Dropout!")
                
        # === VISUAL FEATURE TIME ATTENTION ENCODER ===
        for block in self.visual_blocks:
            visual_tokens = block(visual_tokens)
            
        # === IMU TIME ATTENTION ENCODER ===
        for block in self.imu_blocks:
            imu_tokens = block(imu_tokens)
        
        # === CROSS-MODAL FUSION ===
        for cross_block in self.cross_modal_blocks:
            visual_tokens, imu_tokens = cross_block(visual_tokens, imu_tokens)
            
        # === FUSION OF VISUAL AND IMU TOKENS ===
        # trans_feat, rot_feat = self.fusion_strat(visual_tokens, imu_tokens)
        fused_tokens = self.fusion_strat(visual_tokens, imu_tokens)
        
        # === FINAL NETWORK HEAD FOR PREDICTION ===
        # trans_predictions = self.network_head_trans(self.final_norm_trans(trans_feat))
        # rot_predictions = self.network_head_rot(self.final_norm_rot(rot_feat))
        trans_predictions = self.network_head_trans(self.final_norm_trans(fused_tokens))
        rot_predictions = self.network_head_rot(self.final_norm_rot(fused_tokens))
        hitch_predictions = self.network_head_hitch(self.final_norm_hitch(hitch_tokens))
        
        # reshape to 2D
        trans_predictions = trans_predictions.view(-1, 2, self.num_deltas)
        rot_predictions = rot_predictions.view(-1, 1, self.num_deltas)
        hitch_predictions = hitch_predictions.view(-1, 1, self.num_deltas)
        
        return trans_predictions, rot_predictions, hitch_predictions

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
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: [B, T_visual, D]
        """
        # Temporal self-attention
        x_in = x
        x = self.norm1(x)
        x, _ = self.temporal_attn(x, x, x)
        x = x_in + x
        # x = self.norm1(x)
        
        # MLP
        x_in = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_in + x
        # x = self.norm2(x)
        
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
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: [B, T_imu, D]
        """
        # Temporal self-attention
        x_in = x
        x = self.norm1(x)
        x, _ = self.temporal_attn(x, x, x)
        x = x_in + x
        # x = self.norm1(x)
        
        # MLP
        x_in = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x_in + x
        # x = self.norm2(x)
        
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
        
        # small MLPs
        self.visual_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.imu_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
                
        self.visual_norm1 = nn.LayerNorm(embed_dim)
        self.visual_norm2 = nn.LayerNorm(embed_dim)
        self.imu_norm1 = nn.LayerNorm(embed_dim)
        self.imu_norm2 = nn.LayerNorm(embed_dim)
        
        
    def forward(self, visual_tokens, imu_tokens):
        """Forward block for cross attention

        Args:
            visual_tokens (torch.tensor): Tokens from images [B, T*N_patches, D]
            imu_tokens (torch.tensor): Tokens from IMU [B, T, D]
        Returns:
            fused_tokens (torch.tensor): Fused tokens via cross attention [B, combined_seq_len, D]
        """
        
        # === CROSS-MODAL ATTENTION ===
        
        # save off input for residual connection
        visual_tokens_in = visual_tokens
        imu_tokens_in = imu_tokens

        # normalize tokens
        visual_tokens_norm = self.visual_norm1(visual_tokens)
        imu_tokens_norm = self.imu_norm1(imu_tokens)
        
        # Visual attending to IMU (what IMU info is relevant for visual features)
        visual_tokens, _ = self.visual_to_imu_attn(
            query=visual_tokens_norm,
            key=imu_tokens_norm,
            value=imu_tokens_norm
        )
        visual_tokens = visual_tokens + visual_tokens_in
        
        # run through MLP
        visual_tokens_in = visual_tokens
        visual_tokens = self.visual_norm2(visual_tokens)
        visual_tokens = self.visual_head(visual_tokens)
        visual_tokens = visual_tokens + visual_tokens_in
                
        # IMU attending to visual (what visual info is relevant for IMU features)
        imu_tokens, _ = self.imu_to_visual_attn(
            query=imu_tokens_norm,
            key=visual_tokens_norm,
            value=visual_tokens_norm
        )
        imu_tokens = imu_tokens + imu_tokens_in
        
        # imu MLP
        imu_tokens_in = imu_tokens
        imu_tokens = self.imu_norm2(imu_tokens)
        imu_tokens = self.imu_head(imu_tokens)
        imu_tokens = imu_tokens + imu_tokens_in
        
        return visual_tokens, imu_tokens
        
class GloabalPoolFusionStrategy(nn.Module):
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
        
class TaskSpecificFusionStrategy(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        
        # learnable queries for each task
        self.translation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.rotation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Cross-attention for pooling
        self.translation_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.rotation_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # # small refinement mlps per task
        # self.task_mlp = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim // 2, embed_dim)
        # )
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

    def forward(self, visual_tokens,  imu_tokens):
        B = visual_tokens.shape[0]
        
        # concatenate all tokens
        all_tokens = torch.cat([visual_tokens, imu_tokens], dim=1)
        
        # generate features for each task
        # translation
        trans_feat, _ = self.translation_attention(
            self.translation_query.expand(B, -1, -1),
            all_tokens,
            all_tokens
        )
        trans_feat = trans_feat.squeeze(1)
        trans_feat = self.translation_mlp(trans_feat)
        # rotation
        rot_feat, _ = self.rotation_attention(
            self.rotation_query.expand(B, -1, -1),
            all_tokens,
            all_tokens
        )
        rot_feat = rot_feat.squeeze(1)
        rot_feat = self.rotation_mlp(rot_feat)
        
        return trans_feat, rot_feat