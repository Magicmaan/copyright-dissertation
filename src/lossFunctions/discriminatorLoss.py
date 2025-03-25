"""
Will check if watermark is maintained after NST.
Takes in pre-NST and post-NST images and compares them.

5. Discriminator Loss (What the Discriminator Learns)
Attempts to recognize if watermark is maintained after NST by comparing watermarked image pre-NST and post-NST.

✔ Correct!
🔹 The discriminator is trained to detect if an image has a watermark.
🔹 It compares watermarked images before and after NST.

Equation:

L_D = -log(D(I_s)) - log(1 - D(I_w))

Where:

D(I_s) = Discriminator’s confidence that a pre-NST image has a watermark.
D(I_w) = Discriminator’s confidence that a post-NST image has a watermark.

✔ If D(I_w) is close to 1, NST didn't remove the watermark (bad for NST, good for us!).
✔ If D(I_w) is close to 0, NST removed the watermark (good for NST, bad for us!).
"""
from torch import Tensor
import torch.nn as nn

class DiscriminatorLoss(nn.module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce = nn.BCELoss()