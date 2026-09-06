import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(parent_dir)
import cv2
import io
import imageio
# from vwm.foresight.data import CS_VideoData 
from cosmos_predict2.auxiliary.foresight.dino_f import Dino_f
import argparse
import torch
import pytorch_lightning as pl
import os
import matplotlib.cm as cm
import glob
import os.path as osp
from tqdm import tqdm
import numpy as np
from PIL import Image
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import einops
import math
import torch.nn as nn

def parse_args_dino():
    args = argparse.Namespace()
    # Data Parameters
    args.data_path = '/opt/dlami/nvme/cityscapes/leftImg8bit_sequence'
    args.dst_path = None
    args.img_size = (448,896)
    # args.img_size = (224,224)
    # args.img_size = (224,448)
    args.num_workers = 8
    args.num_workers_val = None
    # args.sequence_length = 3 # kinetics
    args.sequence_length = 5
    args.batch_size = 1
    args.random_crop = True
    args.random_horizontal_flip = True
    args.random_time_flip = False
    args.timestep_augm = None
    args.no_timestep_augm = False
    args.use_fc_bias = True
    args.feature_extractor = 'dino'
    args.dinov2_variant = 'vitb14_reg'
    args.d_layers = [2,5,8,11]
    args.hidden_dim = 1152
    args.heads = 8
    args.layers = 12
    args.dropout = 0.1
    args.loss_type = 'SmoothL1'
    args.beta_smoothl1 = 0.1
    args.attn_dropout = 0.3
    args.step = 1
    args.masking = 'simple_replace'
    args.train_mask_mode = 'fullmask'
    args.seperable_attention = True
    args.seperable_window_size = 1
    args.train_mask_frames = 1
    args.output_activation = 'none'
    args.use_first_last = False
    args.down_up_sample = False
    args.pca_ckpt = "checkpoints/Re2Pix_cs_40k/re2pix_stage1_dinov2_pca_448_l[2_5_8_11]_1152.pth" # Path to the PCA checkpoint
    args.improved_architecture = False
    args.crop_feats = False
    args.sliding_window_inference = False
    args.use_bn = True
    args.use_cls = False
    args.nfeats = 256
    args.dpt_out_channels = [128,256,512,512]
    # Training parameters
    args.max_epochs = 800
    args.seed = 123
    args.single_step_sample_train = True
    args.precision = '32-true'
    args.ckpt = "checkpoints/Re2Pix_cs_40k/re2pix_stage1.ckpt"  # Path to the Re2Pix (ours Dino-Foresight) checkpoint
    args.num_gpus = 1
    args.accum_iter = 1
    args.warmup_p = 0.0
    args.lr_base = 1e-3
    args.weight_decay = 0
    args.scheduler = "cosine"
    args.optimizer = "adam"
    args.gclip = 1.0
    args.evaluate = True
    args.eval_midterm = False
    args.eval_mode = True
    args.use_val_to_train = False
    args.use_train_to_val = False
    args.vis_attn = False
    args.eval_modality = None
    args.high_res_adapt = False
    return args


def parse_args_depth_dpt():
    args = argparse.Namespace()
    # Data Parameters
    args.num_classes = 256
    args.feature_dim = 768 # dinov2_vitb14_reg
    args.nfeats = 256
    args.use_bn = True
    args.dpt_out_channels = [128,256,512,512]
    args.use_cls = False
    return args

def parse_args_segm_dpt():
    args = argparse.Namespace()
    # Data Parameters
    args.num_classes = 19
    args.feature_dim = 768 # dinov2_vitb14_reg
    args.nfeats = 256
    args.use_bn = True
    args.dpt_out_channels = [128,256,512,512]
    args.use_cls = False
    return args

def parse_args_rgb_dpt():
    args = argparse.Namespace()
    # Data Parameters
    args.num_classes = 3
    args.feature_dim = 768 # dinov2_vitb14_reg
    args.nfeats = 256
    args.use_bn = True
    args.dpt_out_channels = [128,256,512,512]
    args.use_cls = False
    return args


def load_dpt_state_dict(model, ckpt_path, requires_grad=False):
    state_dict = {}
    for k, v in torch.load(ckpt_path, weights_only=False)["state_dict"].items():
        state_dict[k.replace("head.","")] = v
    model.load_state_dict(state_dict)
    if not requires_grad:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model