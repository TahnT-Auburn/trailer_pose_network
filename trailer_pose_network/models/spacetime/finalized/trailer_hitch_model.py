import torch
import torch.nn as nn

class HitchModel(nn.Module):
    def __init__(self,
                 encoder:nn.Module,
                 embed_dim,
                 dropout,):
        super().__init__()
        # replace network head
        # self.final_norm_hitch = nn.LayerNorm(embed_dim)
        in_feats = encoder.classifier[1].in_features # classifier[1] for mobilenetv2
        encoder.classifier = nn.Sequential(
            nn.Linear(in_feats, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        self.encoder = encoder
    
    def forward(self, latest_image):
        # === HITCH ENCODER ===
        hitch_prediction = self.encoder(latest_image)
        
        return hitch_prediction