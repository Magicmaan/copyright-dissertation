# Watermark loss, how well watermark is retained / extracted after NST. 
# W = Original Watermark Image
# Ww = Extracted Watermark from NST
# Losswatermark =W -Ww 2

from torch import Tensor
import torch.nn as nn


class WatermarkLoss(nn.module):
    """
        calculate loss between original and extracted watermark using MSE.
    """
    def __init__(self):
        super(WatermarkLoss, self).__init__()
        self.mse: nn.MSELoss = nn.MSELoss()
    
    def forward(self, original_watermark, extracted_watermark) -> Tensor:
        """
            Calculate loss between original watermark and extracted watermark using MSE.
            
            @param: original_watermark: Original watermarked image.
            @param: extracted_watermark: Extracted watermarked image.
            
            :return: Loss between original watermarked image and extracted watermarked image.
        """

        return self.mse(original_watermark, extracted_watermark)
    
