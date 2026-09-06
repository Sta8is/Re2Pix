import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import einops

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vitb14_reg", target_height=576, target_width=1024, pca_ckpt="/home/ubuntu/hieracosmos2/checkpoints/dino_pca_448_l[2,_5,_8,_11]_1152comb_data.pth"):
        super().__init__()
        # Load model directly from torch.hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()  # Set to evaluation mode
        self.emb_dim = self.model.embed_dim
        self.f_h = target_height // 8  # Assuming the VAE outputs at 1/8th of the input size
        self.f_w = target_width // 8 # Assuming the VAE outputs at 1/8th of the input size
        self.pca_dict = torch.load(pca_ckpt, weights_only=False)
        self.pca = self.pca_dict['pca_model']
        self.pca_mean = torch.nn.Parameter(torch.tensor(self.pca.mean_), requires_grad=False)
        self.pca_components = torch.nn.Parameter(torch.tensor(self.pca.components_), requires_grad=False)
        self.mean = torch.nn.Parameter(torch.tensor(self.pca_dict['mean']), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(self.pca_dict['std']),requires_grad=False)
        self.embedding_dim = self.pca_components.shape[0]
        # self.mins_maxes = torch.load("/home/ubuntu/hieracosmos2/checkpoints/mins_maxes.pth", weights_only=False)
        # self.mins = torch.nn.Parameter(torch.tensor(self.mins_maxes['mins']), requires_grad=False)
        # self.maxes = torch.nn.Parameter(torch.tensor(self.mins_maxes['maxes']), requires_grad=False)
    
        # Freeze the model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def pca_transform(self, x):
        BT, HW, C = x.shape
        x = (x - self.mean) / self.std
        x = x - self.pca_mean
        x_pca = torch.matmul(x, self.pca_components.T)
        return x_pca

    def pca_inverse_transform(self, x):
        BT, HW, C = x.shape
        x = torch.matmul(x, self.pca_components) + self.pca_mean
        x = x * self.std + self.mean
        return x

    
            
    def forward(self, images):
        """
        Extract DINOv2 patch features from images
        
        Args:
            images: tensor of shape [B*T, C, H, W]
                B: batch size (always 1), T: frames, C: channels, H: height, W: width
                
        Returns:
            features: tensor of shape [B, T, 768, h, w]
                    where h, w are the spatial dimensions of patch features
        """
        # Input is already [B*T, C, H, W] as specified
        B_times_T, C, H, W = images.shape
        T = B_times_T  # Since batch size is always 1
                
        # Extract features using get_intermediate_layers
        with torch.no_grad():
            # features = self.model.get_intermediate_layers(images, 
            #                                             n=1,  # Get only one layer (the last one)
            #                                             reshape=False)[0]  # Get the first (and only) element in the list
            # features = self.model.get_intermediate_layers(images, [8,11], return_class_token=False, norm=True)
            features = self.model.get_intermediate_layers(images, [2,5,8,11], return_class_token=False, norm=True)
            features = torch.cat(features, dim=-1)  # Concatenate along the channel dimension
            features = self.pca_transform(features)
            # features = features[:,:,:16] 
            # for specific layer features
            # features = self.pca_inverse_transform(features)
            # features = features[:,:,-768:] # last layer features
            # Calculate patch dimensions
            h = int(H // 14)  # ViT-B/14 has patch size 14
            w = int(W // 14)
            
            # Reshape to [B*T, 768, h, w]
            patch_features = einops.rearrange(features, 'bt (h w) c -> bt c h w', h=h, w=w)
            
            # Interpolate to match expected dimensions
            patch_features = F.interpolate(patch_features, 
                                        size=(self.f_h, self.f_w),  # Adjust size as needed
                                        mode='bilinear', 
                                        align_corners=False)

            
            
        return patch_features