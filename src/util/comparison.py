import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision
import matplotlib.pyplot as plt
import math
from torchmetrics.regression import MeanSquaredError
from util.vgg19 import VGG, extractFeatures


def pixelDifference(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the pixel difference between two images.
    Simple pixel-wise operation.
    Args:
        image1: The first image.
        image2: The second image.
    Returns:
        The pixel difference between the two images.
        (lower is better, 0 is perfect match)
    """
    return abs(image1 - image2) / 255.0  # Normalize to [0, 1] range


def MSEDifference(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    Args:
        image1: The first image.
        image2: The second image.
    Returns:
        The MSE between the two images.
        (lower is better, 0 is perfect match)
    """
    assert image1 is not None and image2 is not None, "Both images must be provided"
    assert isinstance(image1, torch.Tensor) and isinstance(
        image2, torch.Tensor
    ), "Inputs must be torch tensors"

    assert (
        image1.shape == image2.shape
    ), "Images must have the same shape after resizing"

    MSE = MeanSquaredError(num_outputs=image1.shape[0], dist_sync_on_step=False)
    return MSE(image1, image2)  # Convert to Python float


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG().to(device).eval()


def perceptualDifference(
    image1: torch.Tensor, image2: torch.Tensor, displayFeatures: bool = False
) -> float:
    """
    Calculate the perceptual difference between two images feature maps using MSE.
    Args:
        image1: The first image.
        image2: The second image.
        displayFeatures: Whether to display the feature maps of the images.
    Returns:
        float: The perceptual difference between the two images.
        (higher is worse, 0 is perfect match)
    """
    image1 = image1.clone().to(device)
    image2 = image2.clone().to(device)
    convLayers = list(range(29))  # Include all layers

    if displayFeatures:
        print("Extracting Feature Maps")

        # Extract feature maps from img1
        images = extractFeatures(vgg, image1, convLayers)
        numFeatures = len(images)
        cols = 4
        rows = math.ceil(numFeatures / cols)
        fig = plt.figure(figsize=(cols * 5, rows * 5))
        for i in range(numFeatures):
            a = fig.add_subplot(rows, cols, i + 1)
            imgplot = plt.imshow(images[i])
            a.axis("off")
            a.set_title(f"Image1 Layer {convLayers[i]}")
        plt.show()

        # Extract feature maps from img2
        images2 = extractFeatures(vgg, image2, convLayers)
        numFeatures2 = len(images2)
        rows2 = math.ceil(numFeatures2 / cols)
        fig = plt.figure(figsize=(cols * 5, rows2 * 5))
        for i in range(numFeatures2):
            a = fig.add_subplot(rows2, cols, i + 1)
            imgplot = plt.imshow(images2[i])
            a.axis("off")
            a.set_title(f"Image2 Layer {convLayers[i]}")
        plt.show()

    # Ensure feature maps are tensors
    image1Features = torch.cat([f.flatten() for f in vgg(image1)], dim=0)
    image2Features = torch.cat([f.flatten() for f in vgg(image2)], dim=0)

    mse = torch.mean((image1Features - image2Features) ** 2).item()
    return mse


ssim = StructuralSimilarityIndexMeasure()


# https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
def structuralDifference(image1, image2) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    Args:
        image1: The first image.
        image2: The second image.
    Returns:
        float: The SSIM between the two images.
        (higher is better, 1 is perfect match)
    """
    return ssim(image1, image2)


def PSNR(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        image1: The first image.
        image2: The second image.
    Returns:
        float: The PSNR between the two images.
    """
    mse = MSEDifference(image1, image2)
    if mse == 0:
        return float("inf")  # No noise, perfect match
    max_pixel_value = 255.0
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()  # Convert to Python float
