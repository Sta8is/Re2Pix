import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import einops
from time import time
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
import math
from cosmos_predict2.auxiliary.foresight.attention_masked import MaskTransformer


class Dino_f(pl.LightningModule):
    def __init__(self,args):
        super(Dino_f,self).__init__()
        self.args = args
        self.improved_architecture = args.improved_architecture
        self.sequence_length = args.sequence_length # 4
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.loss_type = args.loss_type
        self.img_size  = args.img_size
        self.d_layers = args.d_layers
        self.patch_size = 14 if self.args.feature_extractor in ['dino', 'eva2-clip'] else 16
        self.d_num_layers = len(self.d_layers) if isinstance(self.d_layers, list) else self.d_layers
        if not self.args.crop_feats and not self.args.sliding_window_inference:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size), self.img_size[1]//(self.patch_size))
        else:
            self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size*2), self.img_size[1]//(self.patch_size*2))
        if self.args.feature_extractor == 'dino':
            self.dino_v2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_'+args.dinov2_variant, pretrained=True)
            for param in self.dino_v2.parameters():
                param.requires_grad = False
            self.dino_v2.eval()
        elif self.args.feature_extractor == 'eva2-clip':
            self.eva2clip = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in1k', pretrained=True, img_size = (self.img_size[0], self.img_size[1])) # pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])}
            for param in self.eva2clip.parameters():
                param.requires_grad = False
            self.eva2clip.eval()
        elif self.args.feature_extractor == 'sam':
            self.sam = timm.create_model('timm/samvit_base_patch16.sa1b', pretrained=True,  pretrained_cfg_overlay={'input_size': (3,self.img_size[0],self.img_size[1])})
            for param in self.sam.parameters():
                param.requires_grad = False
                self.sam.eval()
        if self.args.feature_extractor == 'dino':
            self.feature_dim = self.dino_v2.embed_dim
        elif self.args.feature_extractor == 'eva2-clip':
            self.feature_dim = self.eva2clip.embed_dim
        elif self.args.feature_extractor == 'sam':
            self.feature_dim = self.sam.embed_dim
        self.embedding_dim = self.d_num_layers * self.feature_dim
        if self.args.pca_ckpt:
            self.pca_dict = torch.load(self.args.pca_ckpt, weights_only=False)
            self.pca = self.pca_dict['pca_model']
            self.pca_mean = torch.nn.Parameter(torch.tensor(self.pca.mean_), requires_grad=False)
            self.pca_components = torch.nn.Parameter(torch.tensor(self.pca.components_), requires_grad=False)
            self.mean = torch.nn.Parameter(torch.tensor(self.pca_dict['mean']), requires_grad=False)
            self.std = torch.nn.Parameter(torch.tensor(self.pca_dict['std']),requires_grad=False)
            self.embedding_dim = self.pca_components.shape[0]
        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
                                       seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size, use_first_last=args.use_first_last)
        self.train_mask_frames = args.train_mask_frames
        self.train_mask_mode = args.train_mask_mode
        self.masking = args.masking
        assert self.masking in ("half_half", "simple_replace", "half_half_previous")
        if self.masking in ("half_half", "half_half_previous"): # default 
            self.mask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.unmask_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            if self.masking=="half_half":
                self.replace_vector = torch.nn.Parameter(torch.randn(1, 1, 1, 1, self.hidden_dim//2))
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim//2)
        elif self.masking == "simple_replace":
            self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
            self.replace_vector = nn.Parameter(torch.zeros(1, 1, 1, 1, self.hidden_dim))
            torch.nn.init.normal_(self.replace_vector, std=.02)
            self.maskvit.fc_in = nn.Identity()
        self.activation = nn.Sigmoid() if args.output_activation == "sigmoid" else nn.Identity()
        # Necessary for evaluation
           
       

    def extract_features(self, x, reshape=False):
        with torch.no_grad():
            if self.args.feature_extractor == 'dino':
                x = self.dino_v2.get_intermediate_layers(x,n=self.d_layers, reshape=reshape)
            elif self.args.feature_extractor == 'eva2-clip':
                x = self.eva2clip.forward_intermediates(x, indices=self.d_layers, output_fmt  = 'NLC', norm=True, intermediates_only=True)
            elif self.args.feature_extractor == 'sam':
                x = self.sam.forward_intermediates(x, indices=self.d_layers, norm=False, intermediates_only=True) # Norm is False to avoide neck layer that reduces feature_dim to 256. Also output is in NCHW format
                x = [einops.rearrange(f, 'b c h w -> b (h w) c') for f in x]
            if self.d_num_layers > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x
            
    def get_mask_tokens(self, x, mode="arccos", mask_frames=1):
        B, sl, h, w, c = x.shape # x.shape [B,T,H,W,C]
        assert mask_frames <= sl
        if mode == "full_mask":
            if self.sequence_length == 7:
                assert mask_frames <= 3
            else:
                assert mask_frames == 1
            mask = torch.ones(B,mask_frames, h,w, dtype=torch.bool)
        else:
            r = torch.rand(B) # Batch size
            if mode == "linear":                # linear scheduler
                val_to_mask = r
            elif mode == "square":              # square scheduler
                val_to_mask = (r ** 2)
            elif mode == "cosine":              # cosine scheduler
                val_to_mask = torch.cos(r * math.pi * 0.5)
            elif mode == "arccos":              # arc cosine scheduler
                val_to_mask = torch.arccos(r) / (math.pi * 0.5)
            else:
                val_to_mask = None
            # Create a mask of size [Batch,1,Height, Width] for the last frame of each sequence
            mask = torch.rand(size=(B, mask_frames, h, w)) < val_to_mask.view(B, 1, 1, 1)

        # Create the mask for all frames, by concatenating the mask for the last frame to a tensor of zeros(no mask) for first T-1 frames
        mask = torch.cat([torch.zeros(B,sl-mask_frames,h,w).bool(), mask], dim=1).to(x.device)

        if self.masking in ("half_half", "half_half_previous"):
            # Create the mask_tokens tensor
            mask_tokens = mask.unsqueeze(-1).float()*self.mask_vector.expand(B,sl,h,w,-1) + (~mask.unsqueeze(-1)).float()*self.unmask_vector.expand(B,sl,h,w,-1)
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            if self.masking == "half_half_previous":
                # Replace the embedded tokens at masked locations with the embedded tokens from the previous frames
                replace = torch.cat((torch.zeros((B,1,h,w,embedded_tokens.shape[-1]), dtype=embedded_tokens.dtype, device=embedded_tokens.device), embedded_tokens[:,:-1]), dim=1)
            else:
                # Replace the embedded tokens at masked locations with the replace vector
                replace = self.replace_vector.expand(B,sl,h,w,-1)
            embedded_tokens = torch.where(mask.unsqueeze(-1), replace, embedded_tokens)
            # Concatenate the masked tokens to the embedded tokens. Only take half of each to get the right hidden size
            final_tokens = torch.cat((embedded_tokens,mask_tokens), dim=-1)
        elif self.masking=="simple_replace":
            # Embed the soft tokens
            embedded_tokens = self.embed(x)
            # Replace the embedded tokens at masked locations with the replace vector
            final_tokens = torch.where(mask.unsqueeze(-1), self.replace_vector.expand(B,sl,h,w,-1), embedded_tokens)

        return final_tokens, mask



    def pca_transform(self, x):
        BT, HW, C = x.shape
        x = (x - self.mean) / self.std
        x = x - self.pca_mean
        x_pca = torch.matmul(x, self.pca_components.T)
        return x_pca

    def pca_inverse_transform(self, x):
        B, T, H, W, C = x.shape
        x = torch.matmul(x, self.pca_components) + self.pca_mean
        x = x * self.std + self.mean
        return x


    def preprocess(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W]. 
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1) # x.shape [B*T,C,H,W]
        x = self.extract_features(x) # [B*T,H*W,C]
        if self.args.pca_ckpt:
            x = self.pca_transform(x)
        x = einops.rearrange(x, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        x = x.unflatten(dim=0, sizes=(B, self.sequence_length)) # [B,T,H,W,C]
        return x

    def postprocess(self, x):
        if self.args.pca_ckpt:
            x = self.pca_inverse_transform(x)
        return x
    
    def calculate_loss(self, x_pred, x_target):
        if self.args.loss_type == "MSE":
            loss = F.mse_loss(x_pred, x_target)
        elif self.args.loss_type == "SmoothL1":
            loss = F.smooth_l1_loss(x_pred, x_target, beta=self.args.beta_smoothl1)
        elif self.args.loss_type == "L1":
            loss = F.l1_loss(x_pred, x_target)
        return loss

    def forward_loss(self, x_pred, x_target, mask):
        B, T, H, W, C = x_pred.shape
        x_target = x_target[mask] 
        x_pred = x_pred[mask] 
        return self.calculate_loss(x_pred, x_target)
        
    def forward(self, x, masked_x, mask=None):
        if self.args.vis_attn:
            x_pred, attn = self.maskvit(masked_x,return_attn=True)
        else:
            x_pred, = self.maskvit(masked_x)
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        x_pred = self.activation(x_pred)
        # if self.args.predict_residuals:
        #     x_prev = torch.cat([torch.zeros(B,1,H,W,C).to(x.device), x[:,:-1]], dim=1)
        #     x_pred = x_pred + x_prev
        loss = self.forward_loss(x_pred=x_pred, x_target=x, mask=mask)
        if self.args.vis_attn:
            return loss, x_pred, attn
        else:
            return loss, x_pred

    