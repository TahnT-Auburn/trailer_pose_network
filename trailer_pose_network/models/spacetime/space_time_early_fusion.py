'''
Transformer Model that uses space-time encoders for early fusion of camera, CAN, and IMU measurements.

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

#%%
class PatchEmbed(nn.Module):
    '''
    Patches an input x of shape (B, T, C, H, W) and embeds them.
    
    Arguments:
        img_size (int): 
            Image size. Currently supports a square image input.
        patch_size (int):
            Patch size.
        in_channels (int):
            Number of channels per image.
        embed_dim (int):
            Embedding dimensions.
    '''
    def __init__(self, inp_size:tuple, patch_size:int, in_channels:int, embed_dim:int):
        super().__init__()
        assert inp_size[0]*inp_size[1] % patch_size == 0, \
            f"Input image size ({inp_size,inp_size}) and patch size ({patch_size}) are not compatible."
        
        patch_size = (patch_size, patch_size)
        num_patches = (inp_size[0]*inp_size[1] // patch_size[0]**2)
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # linear projection from patch space to embedding space using 2D convolution
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.proj(x)
        # W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class Attention(nn.Module):
    '''
    Standard attention block. Applicable for both single and multi-head attention.
    
    Arguments:
        embed_dim (int):
            Embedding dimension.
        num_heads (int):
            Number of attention heads. >1 means multi-head attention.
        qkv_bias (bool):
            Bias factor for learned Q,K,V matrices (FC bias).
        attn_drop (float):
            Dropout factor for attention mechanism. Default is 0.
        proj_drop (float):
            Dropout factor for projection layer. Default is 0.
    '''
    def __init__(self, embed_dim:int, num_heads:int, qkv_bias=False, attn_drop:float=0., proj_drop:float=0.):
        super().__init__()
        self.num_heads = num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SpaceTimeTransformerBlock(nn.Module):
    '''
    Space-Time attention block. Applies the space-time attention mechansim.
    
    Arguments:
        num_frames (int):
            Number frames.
        num_patches (int):
            Number of patches per frame.
        embed_dim (int):
            Embedding dimension.
        num_heads (int):
            Number of attention heads. >1 means multi-head attention.
        qkv_bias (bool):
            Bias factor for learned Q,K,V matrices (FC bias).
        attn_drop (float):
            Dropout factor for attention mechanism. Default is 0.
        proj_drop (float):
            Dropout factor for projection layer. Default is 0.
    '''
    def __init__(self, num_frames:int, num_patches:int, embed_dim:int, num_heads:int, qkv_bias:bool=False, attn_drop:float=0., proj_drop:float=0.):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)

        self.spatial_attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.temporal_attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.temporal_fc = nn.Linear(embed_dim, embed_dim)

        self.mlp = nn.Sequential( #TODO: can make into a separate class with more knobs to turn
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        B = x.shape[0]
        D = x.shape[2]
        N = self.num_patches
        T = self.num_frames

        # temporal attention
        xt = x[:,1:,:] # (B, N*T, D)
        xt = xt.reshape(B, N, T, D).reshape(B*N, T, D)
        xt = self.temporal_attn(self.temporal_norm(xt))
        xt = xt.reshape(B, N, T, D).reshape(B, N*T, D)
        xt = self.temporal_fc(xt)
        xt = xt + x[:,1:,:] # (B, N*T, D)

        # spatial attention
        init_out_token = x[:,0,:].unsqueeze(1)
        out_token = init_out_token.repeat(1, T, 1) # (B, T, D)
        out_token = out_token.reshape(B*T, D).unsqueeze(1)
        xs = xt # (B, N*T, D)
        xs = xs.reshape(B, N, T, D).permute(0, 2, 1, 3).reshape(B*T, N, D)
        xs = torch.cat((out_token, xs), dim=1) # (B*T, N+1, D)
        xs = self.spatial_attn(self.spatial_norm(xs)) 

        # process out token
        out_token = xs[:,0,:] # (B*T, D)
        out_token = out_token.reshape(B, T, D)
        out_token = torch.mean(out_token,dim=1,keepdim=True) # averaging for every frame
        xs = xs[:,1:,:] # (B*T, N, D)
        xs = xs.reshape(B, T, N, D).reshape(B, T*N, D)

        # combine temporal and spatial
        x = torch.cat((init_out_token, xt), dim=1) + torch.cat((out_token, xs), dim=1)

        # pass through transformer MLP with residual connection
        x = x + self.mlp(self.mlp_norm(x))

        return x

class SpaceTimeEncoderEarlyFusion(nn.Module):
    def __init__(self, num_frames:int=8, embed_dim:int=384,
                    img_size:tuple=(224,224), img_patch_size:int=16, img_channels:int=3,
                    inp_size:tuple=(1,4), inp_patch_size:int=1, inp_channels:int=1,
                    num_heads:int=8, depth:int=12, attn_drop:float=0., proj_drop:float=0.):
        super().__init__()
        self.num_frames = num_frames
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        self.img_channels = img_channels
        self.embed_dim = embed_dim
        self.inp_size = inp_size
        self.inp_patch_size = inp_patch_size
        self.inp_channels = inp_channels
        self.depth = depth

        # get embeddings for images
        self.img_patch_embed = PatchEmbed(
            inp_size=img_size,
            patch_size=img_patch_size,
            in_channels=img_channels,
            embed_dim=embed_dim
        )

        # get embeddings for CAN/IMU
        self.inp_patch_embed = PatchEmbed(
            inp_size=inp_size,
            patch_size=inp_patch_size,
            in_channels=inp_channels,
            embed_dim=embed_dim

        )

        self.num_img_patches = self.img_patch_embed.num_patches
        self.num_inp_patches = self.inp_patch_embed.num_patches
        self.num_patches = self.num_img_patches + self.num_inp_patches

        # initialize positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) # NOTE: can also start as zeros
        
        # initialize time embeddings
        self.time_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        # initialize out token (similar to BERT's class token)
        self.out_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # create a module list of space-time transformer blocks
        self.blocks = nn.ModuleList([SpaceTimeTransformerBlock(num_frames=num_frames,
                                                               num_patches=self.num_patches,
                                                               embed_dim=embed_dim,
                                                               num_heads=num_heads,
                                                               attn_drop=attn_drop,
                                                               proj_drop=proj_drop)
                                    for _ in range(self.depth)])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # network head
        #TODO: Make separate from class for more controlability
        self.network_head = nn.Sequential( 
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2)

        )
    def forward(self, inputs):
        
        # let the input be a list with inputs[0] = image and inputs[1] = CAN/IMU
        img = inputs[0]
        inp = inputs[1]

        B, T, C, H, W = img.shape # NOTE: same batchsize and temporal value of CAN and IMU inputs
        N = self.num_patches
        D = self.embed_dim
        
        # tokenize
        img = self.img_patch_embed(img)
        inp= self.inp_patch_embed(inp)

        # early fusion concatenation
        x = torch.cat((img,inp),dim=1)

        out_tokens = self.out_tokens.expand(x.size(0), 1, -1)
        x = torch.cat((out_tokens, x), dim=1)
        x = x + self.pos_embedding #x:(B*T, N+1, D) 
        out_tokens = x[:B, 0, :].unsqueeze(1) # (B, 1, D)
        x = x[:, 1:] # (B*T, N, D)
        x = x.reshape(B, T, N, D).permute(0, 2, 1, 3).reshape(B*N, T, D)
        x = x + self.time_embedding
        x = x.reshape(B, N, T, D).reshape(B, N*T, D)
        x = torch.cat((out_tokens, x), dim=1) # (B, (N*T)+1, D)

        # pass tokens through transformer blocks
        for block in self.blocks:
            x = block(x)

        # perform final normalization
        x = self.final_norm(x)

        # pass output token only through network head
        out = self.network_head(x[:,0])

        return out

