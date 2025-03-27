# Adversarial loss, optimises to embed watermark within NST features. Adds second function to force generator to optimise for NST embedding
# D(Is) = Confidence of watermark being present in NST image
# LossADU = -log(D ( Is) )

# https://medium.com/analytics-vidhya/understanding-gans-deriving-the-adversarial-loss-from-scratch-ccd8b683d7e2


import math
from torch import Tensor
import torch.nn as nn

# TODO
class AdversarialLoss(nn.Module):
    """
        Adversarial loss, optimises to embed watermark within NST features.
        Attempts to fool totalLoss by detecting probability of watermark being present in NST image.
        a high value means the watermark is present (and obvious)
        a low value means the watermark is not present (or not obvious)
        totalLoss will attempt to maximise this loss to make watermark more visible
        this loss will attempt to minimise this loss to make watermark more hidden
    """
    
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCELoss()

    
    def forward(self, styled_image) -> float :
        """
            Calculate loss between original watermarked image and NST watermarked image using MSE.
            
            @param: original_image: Original watermarked image before NST.
            @param: styled_image: Styled image after NST
            
            :return: probability of watermark being present in NST image. (higher means more visible / likely)
            
        """

        return 0.0
        # return self.bce(styled_image)

    
