# Adversarial loss, optimises to embed watermark within NST features. Adds second function to force generator to optimise for NST embedding
# D(Is) = Confidence of watermark being present in NST image
# LossADU = -log(D ( Is) )

# https://medium.com/analytics-vidhya/understanding-gans-deriving-the-adversarial-loss-from-scratch-ccd8b683d7e2


import math
from torch import Tensor
import torch.nn as nn
import torch


# TODO
def adversarialLoss(discriminator, image):
    return -torch.log(discriminator(image)).mean()
