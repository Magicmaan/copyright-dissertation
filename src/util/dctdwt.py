from scipy.fft import dct
from scipy.fft import idct
from torch import Tensor
import torch
from pywt import dwt2, idwt2
import torchvision.transforms as transforms
import numpy as np


def dwt_torch(image) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Perform Discrete Wavelet Transform (DWT) on a tensor image.

    :param: image: Image tensor of shape (batch_size, channels, height, width).

    :return: Tuple of tensors (LL, LH, HL, HH) representing the DWT coefficients.
    """
    # Get shape information
    batch_size, channels, height, width = image.shape

    # Initialize outputs with the same channel dimensions
    LL_list, LH_list, HL_list, HH_list = [], [], [], []

    # Process each channel separately
    for c in range(channels):
        # Extract single channel and convert to numpy
        channel_data = image[:, c : c + 1].squeeze(0).squeeze(0).numpy()

        # Apply DWT
        LL, (LH, HL, HH) = dwt2(channel_data, "haar")

        # Collect results
        LL_list.append(torch.tensor(LL).unsqueeze(0))
        LH_list.append(torch.tensor(LH).unsqueeze(0))
        HL_list.append(torch.tensor(HL).unsqueeze(0))
        HH_list.append(torch.tensor(HH).unsqueeze(0))

    # Stack channels back together
    LL_tensor = torch.cat(LL_list, dim=0).unsqueeze(0)
    LH_tensor = torch.cat(LH_list, dim=0).unsqueeze(0)
    HL_tensor = torch.cat(HL_list, dim=0).unsqueeze(0)
    HH_tensor = torch.cat(HH_list, dim=0).unsqueeze(0)

    return LL_tensor, LH_tensor, HL_tensor, HH_tensor


def idwt_torch(LL, LH, HL, HH) -> Tensor:
    """
    Perform Inverse Discrete Wavelet Transform (IDWT) on tensors.

    :param: LL: Low-Low coefficients.
    :param: LH: Low-High coefficients.
    :param: HL: High-Low coefficients.
    :param: HH: High-High coefficients.

    :return: Reconstructed image tensor.
    """
    # Get shape information
    batch_size, channels, height, width = LL.shape

    # Initialize output
    result_channels = []

    # Process each channel separately
    for c in range(channels):
        # Extract data for this channel
        LL_np = LL[:, c].squeeze(0).numpy()
        LH_np = LH[:, c].squeeze(0).numpy()
        HL_np = HL[:, c].squeeze(0).numpy()
        HH_np = HH[:, c].squeeze(0).numpy()

        # Apply IDWT
        channel_result = idwt2((LL_np, (LH_np, HL_np, HH_np)), "haar")

        # Collect result
        result_channels.append(torch.tensor(channel_result).unsqueeze(0))

    # Stack channels back together
    result = torch.cat(result_channels, dim=0).unsqueeze(0)
    return result


def embedWatermarkDWT(image: Tensor, watermark: Tensor, alpha=0.1) -> Tensor:
    """
    embeds watermark into image using Discrete Wavelet Transform (DWT)

    :param: image: Image tensor.
    :param: watermark: Watermark tensor.
    :param: alpha: Scaling factor for watermark.

    :return: Watermarked image tensor.
    """
    # get dwts of image and watermark
    LL, LH, HL, HH = dwt_torch(image)
    LL_w, LH_w, HL_w, HH_w = dwt_torch(watermark)

    # Make watermark match image channels if needed
    if LL.shape[1] != LL_w.shape[1]:
        # If image has more channels than watermark, replicate watermark channels
        if LL.shape[1] > LL_w.shape[1]:
            LL_w = LL_w.repeat(1, LL.shape[1], 1, 1)
            LH_w = LH_w.repeat(1, LH.shape[1], 1, 1)
            HL_w = HL_w.repeat(1, HL.shape[1], 1, 1)
            HH_w = HH_w.repeat(1, HH.shape[1], 1, 1)

    # embed watermark
    LL_embedded = LL + alpha * LL_w * 2
    LH_embedded = LH + alpha * LH_w
    HL_embedded = HL + alpha * HL_w
    HH_embedded = HH + alpha * HH_w

    watermarked = idwt_torch(LL_embedded, LH_embedded, HL_embedded, HH_embedded)
    return watermarked


def extractWatermarkDWT(
    originalImage: Tensor, watermarkedImage: Tensor, alpha=0.1
) -> Tensor:
    """
    Extracts watermark from watermarked image using DWT.

    :param: originalImage: Original image tensor.
    :param: watermarkedImage: Watermarked image tensor.
    :param: alpha: Scaling factor for watermark.

    :return: Extracted watermark tensor.
    """
    # DWT decomposition
    LL_original, LH_original, HL_original, HH_original = dwt_torch(originalImage)
    LL_watermark, LH_watermark, HL_watermark, HH_watermark = dwt_torch(watermarkedImage)

    # get coefficients of watermark
    LL_extracted = (LL_watermark - LL_original) / alpha
    LH_extracted = (LH_watermark - LH_original) / alpha
    HL_extracted = (HL_watermark - HL_original) / alpha
    HH_extracted = (HH_watermark - HH_original) / alpha

    extracted = idwt_torch(LL_extracted, LH_extracted, HL_extracted, HH_extracted)
    return extracted


def embedWatermarkDCT(image: Tensor, watermark: Tensor, alpha=0.1) -> Tensor:
    """
    embeds watermark into image using Discrete Cosine Transform (DCT)

    :param: image: Image tensor.
    :param: watermark: Watermark tensor.
    :param: alpha: Scaling factor for watermark.

    :return: Watermarked image tensor.
    """
    # Get shape information
    batch_size, channels, height, width = image.shape

    # Make watermark match image channels if needed
    if image.shape[1] != watermark.shape[1]:
        if image.shape[1] > watermark.shape[1]:
            watermark = watermark.repeat(1, image.shape[1], 1, 1)

    # Initialize result tensor
    watermarked = torch.zeros_like(image)

    # Process each channel separately
    for c in range(channels):
        # Extract data for this channel
        image_np = image[:, c].squeeze(0).numpy()
        watermark_np = watermark[:, c].squeeze(0).numpy()

        # Apply DCT
        image_dct = dct(image_np)
        watermark_dct = dct(watermark_np)

        # Embed watermark
        watermarked_dct = image_dct + alpha * watermark_dct

        # Apply IDCT
        result = idct(watermarked_dct)

        # Store result
        watermarked[:, c] = torch.tensor(result)

    return watermarked


def extractWatermarkDCT(
    originalImage: Tensor, watermarkedImage: Tensor, alpha=0.1
) -> Tensor:
    """
    Extracts watermark from watermarked image using DCT.

    :param: original_image: Original image tensor.
    :param: watermarked_image: Watermarked image tensor.
    :param: alpha: Scaling factor for watermark.

    :return: Extracted watermark tensor.
    """
    # Get shape information
    batch_size, channels, height, width = originalImage.shape

    # Initialize result tensor
    extracted = torch.zeros_like(originalImage)

    # Process each channel separately
    for c in range(channels):
        # Extract data for this channel
        original_np = originalImage[:, c].squeeze(0).numpy()
        watermarked_np = watermarkedImage[:, c].squeeze(0).numpy()

        # Apply DCT
        original_dct = dct(original_np)
        watermarked_dct = dct(watermarked_np)

        # Extract watermark
        extracted_dct = (watermarked_dct - original_dct) / alpha

        # Apply IDCT
        result = idct(extracted_dct)

        # Store result
        extracted[:, c] = torch.tensor(result)

    return extracted
