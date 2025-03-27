from scipy.fft import dct
from scipy.fft import idct
from torch import Tensor
import torch
from pywt import dwt2, idwt2
import torchvision.transforms as transforms

def dwt_torch(image):
    """Perform Discrete Wavelet Transform (DWT) on a tensor image."""
    # Convert to grayscale if the image has more than one channel
    if image.shape[0] > 1:
        transform = transforms.Grayscale()
        image = transform(image)
    LL, (LH, HL, HH) = dwt2(image.numpy(), 'haar')
    return torch.tensor(LL), torch.tensor(LH), torch.tensor(HL), torch.tensor(HH)

def idwt_torch(LL, LH, HL, HH):
    """Perform Inverse Discrete Wavelet Transform (IDWT) on tensors."""
    return torch.tensor(idwt2((LL.numpy(), (LH.numpy(), HL.numpy(), HH.numpy())), 'haar'))


def embed_watermark_DWT(image:Tensor, watermark:Tensor, alpha=0.1) -> Tensor:
    """
        embeds watermark into image using Discrete Wavelet Transform (DWT)
    """
    # Ensure image and watermark have the same number of channels
    # if image.shape[1] != watermark.shape[1]:
    #     transform = transforms.Grayscale()
    #     image = transform(image)
    #     watermark = transform(watermark)
    
    # get dwts of image and watermark
    LL, LH, HL, HH = dwt_torch(image)
    LL_w, LH_w, HL_w, HH_w = dwt_torch(watermark)
    
    # embed watermark
    # low frequency (LL) contains most of image information so is ignored
    # high frequency (LH, HL, HH) contains less information so is used to embed watermark
    LL_embedded = LL + alpha * LL_w
    LH_embedded = LH + alpha * LH_w
    HL_embedded = HL + alpha * HL_w
    HH_embedded = HH + alpha * HH_w
    
    watermarked = idwt_torch(LL_embedded, LH_embedded, HL_embedded, HH_embedded)
    return watermarked

def extract_watermark_DWT(original_image:Tensor, watermarked_image:Tensor, alpha=0.1) -> Tensor:
    # DWT decomposition
    _, LH_original, HL_original, HH_original = dwt_torch(original_image)
    _, LH_watermark, HL_watermark, HH_watermark = dwt_torch(watermarked_image)
    
    # get coefficients of watermark
    LH_extracted = ( LH_watermark - LH_original ) / alpha
    HL_extracted = ( HL_watermark - HL_original ) / alpha
    HH_extracted = ( HH_watermark - HH_original ) / alpha
    
    extracted = idwt_torch(torch.zeros_like(LH_extracted), LH_extracted, HL_extracted, HH_extracted)
    return extracted

def embed_watermark_DCT(image:Tensor, watermark:Tensor, alpha=0.05) -> Tensor:
    """
        embeds watermark into image using Discrete Cosine Transform (DCT)
    """
    # Ensure image and watermark have the same number of channels
    # if image.shape[1] != watermark.shape[1]:
    #     transform = transforms.Grayscale()
    #     image = transform(image)
    #     watermark = transform(watermark)
    
    # get dct of image and watermark
    image_dct = dct(image.numpy(), axis=2)
    watermark_dct = dct(watermark.numpy(), axis=2)
    
    # embed watermark
    watermarked_dct = image_dct + alpha * watermark_dct
    
    watermarked = torch.tensor(idct(watermarked_dct, axis=2))
    return watermarked

def extract_watermark_DCT(original_image:Tensor, watermarked_image:Tensor, alpha=0.1) -> Tensor:
    # DCT decomposition
    original_image_dct = dct(original_image.numpy(), axis=2)
    watermarked_image_dct = dct(watermarked_image.numpy(), axis=2)
    
    # get coefficients of watermark
    extracted_dct = (watermarked_image_dct - original_image_dct) / alpha
    
    extracted = torch.tensor(idct(extracted_dct, axis=2))
    return extracted



