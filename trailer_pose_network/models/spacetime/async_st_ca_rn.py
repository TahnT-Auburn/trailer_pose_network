import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import torchvision

class AsyncSpaceTimeCrossAttentionResNet(nn.Module):
    def __init__(
        self,
        img_size = (224,224),
        seqential_lookback = 2,
        in_channels = 3,
        embed_dim = 768,
        num_frames = 2,
        num_imu_samples = 5,
        imu_channels = 6,
        num_heads = 12,
        depth = 12, # TODO: make depth a list which spans all three transformers used
        dropout = 0.,
    ):
        super().__init__()
        assert img_size[0]*img_size[1] % 8 == 0, \
            f"Input image size ({img_size[0],img_size[1]}) are not compatible with ResNet encoder. W,H must be divisible my 8."
            
        self.sequential_lookback = seqential_lookback
        self.num_deltas = self.sequential_lookback - 1
        self.num_frames = num_frames
        self.num_imu_samples = num_imu_samples
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        
        # === IMAGE RESNET ENCODER ===
        # Patch embedding for spatial tokenization
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
        
        # === SPACE-TIME TRANSFORMER BLOCKS FOR IMAGES ===
        self.visual_blocks = nn.ModuleList([
            VisualTemporalBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(depth)
        ])
        
        # === TEMPORAL TRANSFORMER BLOCKS FOR IMU ===
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
        self.fusion_strat = TaskSpecificFusionStrategy(embed_dim, num_heads, dropout)
        
        # === NETWORK HEADS === 
        self.final_norm_trans = nn.LayerNorm(embed_dim)
        self.network_head_trans = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2 * self.num_deltas),
        )

        self.final_norm_rot = nn.LayerNorm(embed_dim)
        self.network_head_rot = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1 * self.num_deltas),
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
        
        # img1 = images[:,0]
        # img2 = images[:,1]
        # feat1 = self.resnet_encoder(img1) #[B, D]
        # feat2 = self.resnet_encoder(img2) #[B, D]
        
        # # stack features
        # visual_tokens = torch.stack([feat1, feat2], dim=1)
        
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
        
        # === EMBEDDING PHASE ===
        visual_tokens = self.embed_images(images)
        imu_tokens = self.embed_imu(imu_data)
        
        # Apply any dropout
        visual_tokens = self.dropout(visual_tokens)
        imu_tokens = self.dropout(imu_tokens)
        
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
        trans_feat, rot_feat = self.fusion_strat(visual_tokens, imu_tokens)
        
        # === FINAL NETWORK HEAD FOR PREDICTION ===
        trans_predictions = self.network_head_trans(self.final_norm_trans(trans_feat))
        rot_predictions = self.network_head_rot(self.final_norm_rot(rot_feat))
        # predictions =  torch.cat((trans_predictions, rot_predictions), dim=1)
        
        # reshape to 2D
        trans_predictions = trans_predictions.view(-1, 2, self.num_deltas)
        rot_predictions = rot_predictions.view(-1, 1, self.num_deltas)
        
        return trans_predictions, rot_predictions

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
        # x = self.norm1(x)
        
        # MLP
        x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
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
        # x = self.norm1(x)
        
        # MLP
        x = self.norm2(x)
        x_mlp = self.mlp(x)
        x = x + x_mlp
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
        visual_tokens = self.visual_norm(visual_tokens)
        visual_attended, _ = self.visual_to_imu_attn(
            query=visual_tokens,
            key=imu_tokens,
            value=imu_tokens
        )
        visual_tokens = visual_tokens + visual_attended
        # visual_tokens = self.visual_norm(visual_tokens)
        
        # IMU attending to visual (what visual info is relevant for IMU features)
        imu_tokens = self.imu_norm(imu_tokens)
        imu_attended, _ = self.imu_to_visual_attn(
            query=imu_tokens,
            key=visual_tokens,
            value=visual_tokens
        )
        imu_tokens = imu_tokens + imu_attended
        # imu_tokens = self.imu_norm(imu_tokens)
        
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
        # self.task_queries = nn.Parameter(torch.randn(3,1, embed_dim))
        self.translation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.rotation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
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


    def forward(self, visual_tokens,  imu_tokens):
        B = visual_tokens.shape[0]
        
        # concatenate all tokens
        all_tokens = torch.cat([visual_tokens, imu_tokens], dim=1)
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

        return trans_feat, rot_feat