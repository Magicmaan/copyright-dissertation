import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision
import matplotlib.pyplot as plt
import math
from torchmetrics.regression import MeanSquaredError
from util.debug import display_image_np, display_image_tensors
from util.vgg19 import VGG


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


def pixel_difference_float(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
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
    return torch.mean(torch.abs(image1 - image2))  # Convert to Python float


from torchmetrics.functional.regression import mean_squared_error


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

    # MSE = MeanSquaredError(num_outputs=image1.shape[0], dist_sync_on_step=False)
    return mean_squared_error(image1, image2)  # Convert to Python float


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG().to(device).eval()


def perceptualDifference(
    image1: torch.Tensor, image2: torch.Tensor, displayFeatures: bool = False
) -> torch.Tensor:
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

    image1_features_np, image1_features_tensor = vgg.extract_features(image1)
    image2_features_np, image2_features_tensor = vgg.extract_features(image2)

    difference: list[torch.Tensor] = []
    for i in range(len(image1_features_tensor)):
        features1 = image1_features_tensor[i] * 10
        features2 = image2_features_tensor[i] * 10

        diff = (features1 - features2) * 100
        difference.append(diff)

    if displayFeatures:
        display_image_np(image1_features_np, convLayers)

        display_image_np(image2_features_np, convLayers)

        display_image_np(
            [tensor.cpu().detach().numpy()[0, 0] for tensor in difference], convLayers
        )

    # get total MSE
    image1Features = torch.cat([f.flatten() for f in image1_features_tensor], dim=0)
    image2Features = torch.cat([f.flatten() for f in image2_features_tensor], dim=0)

    mse = mean_squared_error(image1Features, image2Features)
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


from torchmetrics.functional.image import peak_signal_noise_ratio


def PSNR(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        image1: The first image.
        image2: The second image.
    Returns:
        float: The PSNR between the two images.
    """

    return peak_signal_noise_ratio(image1, image2, data_range=255.0).item()

    # mse = MSEDifference(image1, image2)
    # if mse == 0:
    #     return float("inf")  # No noise, perfect match
    # max_pixel_value = 255.0
    # psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    # return psnr.item()  # Convert to Python float
