# Watermark loss, how well watermark is retained / extracted after NST. 
# W = Original Watermark Image
# Ww = Extracted Watermark from NST
# Losswatermark =W -Ww 2

import torch
import torch.nn as nn


class WatermarkLoss(nn.module):
    def __init__(self):
        super(WatermarkLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, original_watermark, extracted_watermark) -> torch.Tensor:
        """
            Calculate loss between original watermarked image and extracted watermarked image using MSE.
            
            @param: original_watermark: Original watermarked image.
            @param: extracted_watermark: Extracted watermarked image.
            
            :return: Loss between original watermarked image and extracted watermarked image.
        """

        return self.mse(original_watermark, extracted_watermark)
    
