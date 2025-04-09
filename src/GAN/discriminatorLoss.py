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
import torch
import torch.nn as nn


def discriminatorLoss(discriminator, real, fake):
    """
    Discriminator loss function.
    :param discriminator: Discriminator model.
    :param real: Real image tensor (pre-NST).
    :param fake: Fake image tensor (post-NST).

    :return: Discriminator loss.
    aims to minimize the difference between the real and fake images by giving output of 1 for real images and 0 for fake images.
    """
    real_loss = -torch.log(discriminator(real) + 1e-8)  # Want D(Iw) → 1
    fake_loss = -torch.log(1 - discriminator(fake) + 1e-8)  # Want D(Is) → 0
    return real_loss + fake_loss
