"""
    class to compute the perceptual loss between original and watermarked image (before NST)
    done to ensure that the watermark stays visible within watermarked image 
    
    will be done by comparing feature maps of before and after watermark using vgg19.
    
    
    will attempt to ensure feature maps between original and watermarked image are similar.
"""


import torchvision.models as models
from torch import Tensor
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # load vgg19 model with its 16 convolutional layers
        self.vgg = models.vgg19(pretrained=True).features[:16]
        
        self.feature_extractor = nn.Sequential(*list(self.vgg))
        
        self.mse = nn.MSELoss()
    
    def forward(self, original_image: Tensor, watermarked_image: Tensor) -> Tensor:
        """ 
            Calculate perceptual loss between original and watermarked image feature maps
            
            """
            
        # extract feature maps from original and watermarked image
        # will give list of 16 feature maps for each image
        features_1 = self.feature_extractor(original_image)
        features_2 = self.feature_extractor(watermarked_image)
            
        # calculate loss using MSE between feature maps
        return self.mse(features_1, features_2)