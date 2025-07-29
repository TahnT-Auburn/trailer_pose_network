'''
################ Vanilla VIO Transformer ################

A vanilla implementation of a visual-inertial transformer
model. Key idea here is that self attention applies to
embedded patches from FlowNet output and inertial encoder.
Minimal exposure to past measurements.

Author: Tahn Thawainin
        Email: pzt0029@auburn.edu
        github: https://github.com/TahnT-Auburn
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import cv2

from flownet.FlowNetPytorch import models
from flownet.FlowNetPytorch.util import flow2rgb

#%%
class PatchEmbed(nn.Module):
    '''
    Patches an input x of shape (B, C, H, W) and projects
    patches to an embedding space using convolution.
    
    Arguments:
        inp_size (tuple):
            input spatial size (H,W).
        patch_size (tuple):
            patch spatial size.
        in_channels (int):
            Number of channels per input
        embed_dim (int):
            Embedding dimensions.
    '''
    def __init__(self, inp_size:tuple, patch_size:int, in_channels:int, embed_dim:int):
        super().__init__()
        assert inp_size[0]*inp_size[1] % patch_size == 0, \
            f"Input image size ({inp_size[0],inp_size[1]}) and patch size ({patch_size}) are not compatible."
        
        patch_size = (patch_size, patch_size)
        num_patches = (inp_size[0]*inp_size[1] // patch_size[0]**2)
        self.inp_size = inp_size
        self.patch_size = patch_size
        self.in_channels = in_channels 
        self.num_patches = num_patches
        
        # linear projection from patch space to embedding space using convolution.
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels,\
            f"Input Height ({H}) does not have initialized channels ({self.inp_size[0]})"
        assert C == self.in_channels,\
            f"Input Width ({W}) does not have initialized channels ({self.inp_size[1]})"
        assert C == self.in_channels,\
            f"Input Channels ({C}) does not have initialized channels ({self.in_channels})"
        
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2) #(B, NUM_PATCHES, EMBED_DIM)
        return x
    
class FlowNetVisEncoder(nn.Module):
    '''
    FlowNet visual encoder. Uses pre-trained FlowNet for optical flow
    features between two consecutive images.
    
    Arguments:
        weights (str): Path to pretrained FlowNet weights.
        img_size (tuple): Image spatial size (H,W).
        patch_size (int): Patch spatial size.
        embed_dim (int): Embedding dimension.
        device (torch.device): Device ("cpu" or "cuda"). Default is cpu.
    '''
    def __init__(self, weights, img_size:(tuple), patch_size:int, embed_dim:int, device=torch.device("cpu")):
        super().__init__()
        # List any assertions here
        
        self.weights = weights
        img_h, img_w = img_size
        # self.patch_embed = PatchEmbed(inp_size=(img_h//64,img_w//64), # initialized with default FLowNet output shapes and desired patch size and embedding dimensions
        #                               patch_size=patch_size,
        #                               in_channels=1024,
        #                               embed_dim=embed_dim)
        
        # linear projection from FLowNet output to embedding space. #NOTE: Created with default output of FlowNet CONV layers. Will need to modify if extracting different features from FlowNet
        self.lin_proj = nn.Linear(1024,embed_dim)
        # self.lin_proj = nn.Linear(2,embed_dim)

        # initialize learned positional embeddings for visual tokens
        self.vis_pos_embeddings = nn.Parameter(torch.randn(1, img_h//64*img_w//64, embed_dim))
        # self.vis_pos_embeddings = nn.Parameter(torch.randn(1, img_h//4*img_w//4, embed_dim))
        
        # load pretrained model and freeze weights
        network_data = torch.load(weights)
        self.flownet_model = models.__dict__[network_data["arch"]](network_data).to(device)
        for param in self.flownet_model.parameters():
            param.requires_grad = False
            
    def forward(self,x):
        '''
        Arguments:
            x (tensor): Consecutive images concatenated along the channel dimensions. Has shape (B, 2*C, W, H) where C is a single image's channels.
        '''
        with torch.no_grad():
            self.flownet_model.eval()
            flownet_output = self.flownet_model(x)
        
        # flatten flownet output for linear projection
        flownet_output = flownet_output.flatten(2).transpose(2,1)    
        
        # pass flownet output to learned projection for embeddings
        x = self.lin_proj(flownet_output)
        
        # add visual positional embeddings
        x = x + self.vis_pos_embeddings
        
        return x
    
class InertialEncoder(nn.Module):
    '''
    Inertial encoder. The inertial encoder also includes measurements
    from the CAN bus if prompted.
    
    Arguments:
        num_features (int): Number of different inertial measurements
        num_sequence(int): Sequence or temporal number
        embed_dim (int): Embedding dimensions
        drop_out (float): dropout criterion
    '''
    def __init__(self, num_features:int, num_sequence:int, embed_dim:int, drop_out:float=0.):
        super().__init__()
        #NOTE: Assumes IMU/CAN inputs are a batch of vectors (B, INP_SIZE)
        self.num_features = num_features
        self.num_sequence = num_sequence
        self.embed_dim = embed_dim
        self.drop_out = drop_out
        # self.patch_embed = PatchEmbed(inp_size=inp_size,
        #                               patch_size=1,
        #                               in_channels=1,
        #                               embed_dim=embed_dim)
        # self.proj = nn.Linear(self.inp_size, self.embed_dim)
        
        self.encoder_mlp = nn.Sequential(
            nn.Linear(num_sequence*num_features,  64),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(128, embed_dim)
        )
        
    def forward(self,x):
        '''
        Arguments:
            x (input):  Inertial inputs expected to have shape (B, S, N).
                        Where B: batch, S: sequence length, N: number of features (e.g., measurement types)
        '''
        B,S,N = x.shape
        x = x.reshape(B, S*N)
        x = self.encoder_mlp(x)
        x = x.unsqueeze(1)
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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int, proj_drop=0., attn_drop=0.):
        super().__init__()
        
        # layer normalization to normalize the input data
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # call attention block
        self.attention = Attention(embed_dim=embed_dim, num_heads=num_heads, proj_drop=proj_drop, attn_drop=attn_drop)
        
        # post attention layernorm
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # transformer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.LayerNorm(embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
    
    def forward(self, x):

        # apply multihead attention 
        # NOTE: includes first norm layer and a residual connection
        x = self.attention(self.norm1(x)) + x
        
        # apply MLP
        # NOTE: includes second norm layer and a residual connection
        x = self.mlp(self.norm2(x)) + x
        
        return x
    
class VanillaVIOTransformer(nn.Module):
    '''
    Main vanilla VIO transformer module.
    
    Arguments:
        embed_dim (int): Embedding dimensions.
        num_heads (int): Number of heads for multihead attention
        num_layers (int): Number of stacked transformer layers.
        proj_drop (float): Float percentage of dropout for projection layers in the attention mechanism.
        attn_drop (float): Float percentage of dropout for attention layers in teh attention mechanism.
        vis_encoder_params(dict): Dictionary defining neccessary parameters for the visual encoder.
        inert_encoder_params(dict): Dictionary defining neccessary parameters for the inertial encoder.
        num_outputs (int): Number of outputs from the network head.
    '''
    def __init__(self, embed_dim:int, num_heads:int, num_layers:int, proj_drop:float, attn_drop:float, vis_encoder_params:dict, inert_encoder_params:dict, num_outputs:int):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.vis_enc_weights = vis_encoder_params["weights"] # str path
        self.vis_enc_img_size = vis_encoder_params["img_size"] # tuple (H,W)
        self.vis_enc_patch_size = vis_encoder_params["patch_size"] # int
        self.inert_enc_num_features = inert_encoder_params["num_features"] # int
        self.inert_enc_num_sequence = inert_encoder_params["num_sequence"] # int
        self.inert_enc_drop_out = inert_encoder_params["drop_out"] # float
        self.num_outputs = num_outputs
        
        # call visual encoder
        self.vis_encoder = FlowNetVisEncoder(weights=self.vis_enc_weights,
                                             img_size=self.vis_enc_img_size,
                                             patch_size=self.vis_enc_patch_size,
                                             embed_dim=self.embed_dim)
        
        # call inertial encoder
        self.inert_encoder = InertialEncoder(num_features=self.inert_enc_num_features,
                                             num_sequence=self.inert_enc_num_sequence,
                                             drop_out=self.inert_enc_drop_out,
                                             embed_dim=self.embed_dim)
        
        # compute sequence length vis_seq + imu_seq + cls_token
        # self.seq_length = (self.vis_enc_img_size[0]//64*self.vis_enc_img_size[1]//64 // self.vis_enc_patch_size**2) +  1
        
        self.seq_length = (self.vis_enc_img_size[0]//64*self.vis_enc_img_size[1]//64) + 1 + 1
        # self.seq_length = (self.vis_enc_img_size[0]//4*self.vis_enc_img_size[1]//4) + 1 + 1
        # initialize output embedding
        self.out_embeddings = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # initialize global positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_length, embed_dim))
        
        # create core transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim=self.embed_dim,
                             num_heads=self.num_heads,
                             proj_drop=self.proj_drop,
                             attn_drop=self.attn_drop) for _ in range(num_layers)
        ])
        
        # define post transformer layernorm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # define a network head
        self.network_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.Linear(embed_dim*2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_outputs)
        )
    
    def forward(self, inputs):
        '''
        Arguments:
            inputs (list): A list including the the image (processed for FlowNet) at index 0 and the IMU/CAN input at index 1
        '''
        imgs = inputs[0]
        inert = inputs[1]

        # # visualize input images (sanity check)
        # img_iso = imgs[0,:,:,:].reshape(1,2,3,512,512).squeeze(0)
        # img1 = img_iso[0,:,:,:]
        # img2 = img_iso[1,:,:,:]
        # cv2.imshow("img",img1.permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        # cv2.imshow("img2", img2.permute(1, 2, 0).cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # pass inputs to encoders
        vis_embeddings = self.vis_encoder(imgs)
        inert_embeddings = self.inert_encoder(inert)
        
        # concatentate along sequence dim (intermediate fusion)
        x = torch.cat((inert_embeddings, vis_embeddings), dim=1)
        
        # process and concatentat output tokens
        out_tokens = self.out_embeddings.expand(x.size(0), 1, -1)
        x = torch.cat((out_tokens, x), dim=1)
        
        # add positional embeddings
        x = x + self.pos_embedding
        
        # pass fully processed embeddings to transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # apply final layer norm
        x = self.final_norm(x)
        
        # pass output token only to network head
        out = self.network_head(x[:,0])
        
        return out