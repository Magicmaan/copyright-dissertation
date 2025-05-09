# from scipy.fft import dct
# from scipy.fft import idct
from pytorch_wavelets import DWT2D, IDWT2D
from torch_dct import dct, idct
from torch import Tensor
import torch

# from pywt import dwt2, idwt2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


dwt2 = DWT2D(wave="haar", mode="zero", J=3).cuda()
idwt2 = IDWT2D(wave="haar").cuda()
from util.debug import display_image_tensors
from util.vgg19 import VGG, extractFeatures


def dwt_torch(image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Perform Discrete Wavelet Transform (DWT) on a tensor image using PyTorch Wavelets.

    :param image: Image tensor of shape (batch_size, channels, height, width).
    :return: Tuple of tensors (LL, LH, HL, HH) representing the DWT coefficients.
    """
    # Apply DWT - PyTorch Wavelets returns (LL, [high_coeffs])
    LL, high_coeffs = dwt2(image)

    # The high frequency coefficients are returned as a list with
    # each item containing coefficients for a different scale
    # We're interested in the first scale (finest detail)
    high_freq = high_coeffs[0]

    # PyTorch Wavelets stacks LH, HL, HH along dimension 2
    # high_freq shape is (batch, channels, 3, height, width)
    # We need to extract each component
    LH = high_freq[:, :, 0, :, :]  # Horizontal detail
    HL = high_freq[:, :, 1, :, :]  # Vertical detail
    HH = high_freq[:, :, 2, :, :]  # Diagonal detail

    return LL, LH, HL, HH


def idwt_torch(LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor) -> Tensor:
    """
    Perform Inverse Discrete Wavelet Transform (IDWT) on tensors using PyTorch Wavelets.

    :param LL: Low-Low coefficients tensor of shape (batch_size, channels, height, width).
    :param LH: Low-High coefficients tensor of shape (batch_size, channels, height, width).
    :param HL: High-Low coefficients tensor of shape (batch_size, channels, height, width).
    :param HH: High-High coefficients tensor of shape (batch_size, channels, height, width).
    :return: Reconstructed image tensor.
    """
    # Print shapes for debugging
    # print(f"idwt_torch - LL shape: {LL.shape}")
    # print(f"idwt_torch - LH shape: {LH.shape}")
    # print(f"idwt_torch - HL shape: {HL.shape}")
    # print(f"idwt_torch - HH shape: {HH.shape}")

    # Ensure all high-frequency components have the same shape
    target_shape = LH.shape
    if HL.shape != target_shape:
        HL = torch.nn.functional.interpolate(
            HL, size=target_shape[2:], mode="bilinear", align_corners=False
        )
    if HH.shape != target_shape:
        HH = torch.nn.functional.interpolate(
            HH, size=target_shape[2:], mode="bilinear", align_corners=False
        )

    # Stack the high frequency components along dimension 2
    high_freq = torch.stack([LH, HL, HH], dim=2)

    # Create a list with the high frequency components (only one scale in our case)
    high_coeffs = [high_freq]

    # Ensure LL shape is compatible with high_freq for reconstruction
    if LL.shape[2:] != LH.shape[2:]:
        LL = torch.nn.functional.interpolate(
            LL, size=LH.shape[2:], mode="bilinear", align_corners=False
        )

    # print(f"idwt_torch - LL prepared shape: {LL.shape}")
    # print(f"idwt_torch - high_freq shape: {high_freq.shape}")

    # Perform inverse DWT
    reconstructed = idwt2((LL, high_coeffs))

    return reconstructed


def embedWatermarkDWT(image: Tensor, watermark: Tensor, alphas: Tensor) -> Tensor:
    """
    Embeds watermark into image using Discrete Wavelet Transform (DWT).

    :param: image: Image tensor.
    :param: watermark: Watermark tensor.
    :param: alphas: Tensor of scaling factors [LL_alpha, LH_alpha, HL_alpha, HH_alpha].

    :return: Watermarked image tensor.
    """
    assert alphas.numel() == 4, "alphas must be a tensor with 4 scaling factors."
    alphas_list = alphas.tolist() if isinstance(alphas, Tensor) else alphas
    # print("Image shape:", image.shape)
    # print("Watermark shape:", watermark.shape)

    # Make sure watermark matches image spatial dimensions before DWT
    if image.shape[2:] != watermark.shape[2:]:
        watermark = torch.nn.functional.interpolate(
            watermark, size=image.shape[2:], mode="bilinear", align_corners=False
        )

    # Make watermark match image channels if needed
    if image.shape[1] != watermark.shape[1]:
        if image.shape[1] > watermark.shape[1]:
            watermark = watermark.repeat(1, image.shape[1] // watermark.shape[1], 1, 1)

    # print("Adjusted watermark shape:", watermark.shape)

    # Get DWTs of image and watermark
    LL, LH, HL, HH = dwt_torch(image)
    LL_w, LH_w, HL_w, HH_w = dwt_torch(watermark)

    # print("LL shape:", LL.shape, "LL_w shape:", LL_w.shape)
    # print("LH shape:", LH.shape, "LH_w shape:", LH_w.shape)

    # Embed watermark
    # Uses different scaling factors for different frequencies
    LL_embedded = LL + (alphas_list[0] * LL_w)
    LH_embedded = LH + (alphas_list[1] * LH_w)
    HL_embedded = HL + (alphas_list[2] * HL_w)
    HH_embedded = HH + (alphas_list[3] * HH_w)

    # Print shapes before reconstruction
    # print("LL_embedded shape:", LL_embedded.shape)
    # print("LH_embedded shape:", LH_embedded.shape)
    # print("HL_embedded shape:", HL_embedded.shape)
    # print("HH_embedded shape:", HH_embedded.shape)

    # Try using the original pytorch_wavelets approach for reconstruction
    # Stack high frequency components
    high_freq = torch.stack([LH_embedded, HL_embedded, HH_embedded], dim=2)
    high_coeffs = [high_freq]

    # Directly use the pytorch_wavelets IDWT2D
    watermarked = idwt2((LL_embedded, high_coeffs))

    # Clamp the watermarked tensor to ensure values stay within a valid range
    watermarked = torch.clamp(watermarked, 0, 1)

    return watermarked


def extract_watermark_dwt(
    originalImage: Tensor,
    watermarkedImage: Tensor,
    alphas: list[Tensor, Tensor, Tensor, Tensor],
    display=False,
) -> Tensor:
    """
    Extracts watermark from watermarked image using DWT.

    :param: originalImage: Original image tensor.
    :param: watermarkedImage: Watermarked image tensor.
    :param: alphas: Tensor of scaling factors [LL_alpha, LH_alpha, HL_alpha, HH_alpha].
    :param: display: Boolean to display images using matplotlib.

    :return: Extracted watermark tensor.
    """
    assert alphas.__len__() == 4, "alphas must be a list with 4 scaling factors."

    # DWT decomposition
    LL_original, LH_original, HL_original, HH_original = dwt_torch(originalImage)
    LL_watermark, LH_watermark, HL_watermark, HH_watermark = dwt_torch(watermarkedImage)

    # Get coefficients of watermark
    LL_extracted = (LL_watermark - LL_original) / alphas[0]
    LH_extracted = (LH_watermark - LH_original) / alphas[1]
    HL_extracted = (HL_watermark - HL_original) / alphas[2]
    HH_extracted = (HH_watermark - HH_original) / alphas[3]

    amplifiedLL = LL_extracted * 255
    amplifiedLH = LH_extracted * 255
    amplifiedHL = HL_extracted * 255
    amplifiedHH = HH_extracted * 255

    if display:
        display_image_tensors(
            amplifiedLL,
            amplifiedLH,
            amplifiedHL,
            amplifiedHH,
            titles=["Amplified LL", "Amplified LH", "Amplified HL", "Amplified HH"],
        )
    extracted = idwt_torch(LL_extracted, LH_extracted, HL_extracted, HH_extracted)
    return extracted


def embed_watermark_dct(image: Tensor, watermark: Tensor, alpha: Tensor) -> Tensor:
    """
    Embeds watermark into image using Discrete Cosine Transform (DCT).

    :param image: Image tensor.
    :param watermark: Watermark tensor.
    :param alpha: Tensor scaling factor for watermark.

    :return: Watermarked image tensor.
    """
    # Convert tensor alpha to float if needed
    alpha_float = alpha.item()

    # Ensure watermark has the same spatial dimensions as the image
    if image.shape[2:] != watermark.shape[2:]:
        watermark = torch.nn.functional.interpolate(
            watermark, size=image.shape[2:], mode="bilinear", align_corners=False
        )

    # Make watermark match image channels if needed
    if image.shape[1] != watermark.shape[1]:
        if image.shape[1] > watermark.shape[1]:
            watermark = watermark.repeat(1, image.shape[1] // watermark.shape[1], 1, 1)

    # Apply DCT to entire tensors
    # Keep tensors on their original device to maintain gradient flow
    image_device = image.device
    watermark_device = watermark.device

    image_dct = dct(image)
    watermark_dct = dct(watermark)

    # Embed watermark
    watermarked_dct = image_dct + alpha_float * watermark_dct

    # Apply IDCT to get the watermarked image
    watermarked = idct(watermarked_dct)  # Fixed: using idct instead of dct for inverse

    # Clamp values to valid range
    watermarked = torch.clamp(watermarked, 0, 1)

    return watermarked


def extract_watermark_dct(
    original_image: Tensor, watermarked_image: Tensor, alpha: Tensor = torch.tensor(0.1)
) -> Tensor:
    """
    Extracts watermark from watermarked image using DCT.

    :param original_image: Original image tensor.
    :param watermarked_image: Watermarked image tensor.
    :param alpha: Tensor scaling factor for watermark.

    :return: Extracted watermark tensor.
    """
    # Convert tensor alpha to float if needed
    alpha_float = alpha.item() if isinstance(alpha, Tensor) else alpha

    # Ensure images have the same dimensions
    if original_image.shape != watermarked_image.shape:
        watermarked_image = torch.nn.functional.interpolate(
            watermarked_image,
            size=original_image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    # Apply DCT to both images directly using torch-dct
    original_dct = dct(original_image)
    watermarked_dct = dct(watermarked_image)

    # Extract watermark
    extracted_dct = (watermarked_dct - original_dct) / alpha_float

    # Apply IDCT to get the watermark
    extracted = idct(extracted_dct)

    # Clamp values to valid range
    extracted = torch.clamp(extracted, 0, 1)

    return extracted


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = VGG().to(device).eval()


# def experimentalFeatureDWT(
#     image: Tensor, watermark: Tensor, alphasDWT: list[float]
# ) -> list[Tensor]:

#     _, imageFeatures = extractFeatures(vgg, image.cpu(), [0, 5, 10, 19, 28])
#     _, watermarkFeatures = extractFeatures(vgg, watermark.cpu(), [0, 5, 10, 19, 28])

#     finalFeatures = []

#     for i in range(len(imageFeatures)):
#         feature1 = imageFeatures[i].cpu().detach()
#         feature2 = watermarkFeatures[i].cpu().detach()

#         watermarkedFeature = embedWatermarkDWT(feature1, feature2, alphasDWT)

#         # Ensure all tensors have the same shape
#         targetShape = (
#             finalFeatures[0].shape if finalFeatures else watermarkedFeature.shape
#         )
#         watermarkedFeature = torch.nn.functional.interpolate(
#             watermarkedFeature.unsqueeze(0),
#             size=targetShape[2:],
#             mode="bilinear",
#             align_corners=False,
#         ).squeeze(0)

#         finalFeatures.append(watermarkedFeature)

#     finalImage = recombineFeatures(finalFeatures)
#     finalImage = finalImage.to(device).clamp(0, 1)
#     finalImage = finalImage.squeeze(0).cpu()

#     return finalImage


def embedWatermark(
    image: Tensor,
    watermark: Tensor,
    DWT_alphas: list[Tensor, Tensor, Tensor, Tensor],
    DCT_alpha: Tensor = torch.tensor(0.01),
    display: bool = False,
) -> list[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Embeds watermark into image using both DWT and DCT methods.

    :param: image: Image tensor.
    :param: watermark: Watermark tensor.
    :param: DWT_alphas: Tensor of scaling factors for DWT watermark [LL_alpha, LH_alpha, HL_alpha, HH_alpha].
    :param: DCT_alpha: Tensor scaling factor for DCT watermark.
    :param: display: Boolean to display images using matplotlib.

    :return: List of tensors:
        1. Combined watermarked image using DWT and DCT.
        2. Extracted watermark using DWT and DCT.
        3. Watermarked image using DWT.
        4. Watermarked image using DCT.
        5. Extracted watermark using DWT.
        6. Extracted watermark using DCT.
    """
    # assert DWT_alphas.numel() == 4, "DWT_alphas must be a tensor with 4 values."
    assert DWT_alphas.__len__() == 4, "DWT_alphas must be a list with 4 values."

    # Keep tensors on their original device
    device = image.device

    # First embed using DWT (apply DWT watermarking)
    watermarked_dwt = embed_watermark_dwt(image, watermark, DWT_alphas)

    # Then embed using DCT (apply DCT on top of DWT result)
    watermarked_dct = embed_watermark_dct(watermarked_dwt, watermark, DCT_alpha)

    # Extract watermarks to verify quality
    extracted_dwt = extract_watermark_dwt(
        image, watermarked_dwt, DWT_alphas, display=False
    )
    extracted_dct = extract_watermark_dct(watermarked_dwt, watermarked_dct, DCT_alpha)

    # The final extracted watermark is from the DCT step
    final_watermarked = watermarked_dct
    final_extracted = extracted_dct

    if display:
        display_image_tensors(
            image,
            watermark,
            watermarked_dct,
            extracted_dct,
            final_watermarked,
            final_extracted,
            titles=[
                "Content Image",
                "Watermark",
                "Watermarked Image DCT",
                "Extracted Watermark DCT",
                "Final Watermarked Image DCT+DWT",
                "Final Extracted Watermark",
            ],
        )

    return [
        final_watermarked,
        final_extracted,
        watermarked_dwt,
        watermarked_dct,
        extracted_dwt,
        extracted_dct,
    ]


def embed_watermark_dwt(
    image: Tensor, watermark: Tensor, alphas: list[Tensor, Tensor, Tensor, Tensor]
) -> Tensor:
    """
    Embeds watermark into image using Discrete Wavelet Transform (DWT).

    :param: image: Image tensor.
    :param: watermark: Watermark tensor.
    :param: alphas: Tensor of scaling factors [LL_alpha, LH_alpha, HL_alpha, HH_alpha].

    :return: Watermarked image tensor.
    """
    assert alphas.__len__() == 4, "alphas must be a list with 4 scaling factors."
    # alphas_list = alphas.tolist() if isinstance(alphas, Tensor) else alphas

    # Match dimensions before applying DWT
    if image.shape[2:] != watermark.shape[2:]:
        watermark = torch.nn.functional.interpolate(
            watermark, size=image.shape[2:], mode="bilinear", align_corners=False
        )

    if image.shape[1] != watermark.shape[1]:
        watermark = watermark.repeat(1, image.shape[1] // watermark.shape[1], 1, 1)

    # Apply DWT to both image and watermark
    yl_image, yh_image = dwt2(image)

    import matplotlib.pyplot as plt

    def display_dwt_coefficients(yl: Tensor, yh: list[Tensor]) -> None:
        """
        Display the DWT coefficients (LL, LH, HL, HH) using matplotlib.

        :param yl: Low-frequency (LL) coefficients tensor.
        :param yh: List of high-frequency coefficients tensors (LH, HL, HH).
        """
        # Display LL coefficients
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(yl[0, 0].cpu().detach().numpy(), cmap="gray")
        # plt.title("LL Coefficients")
        plt.axis("off")

        # Display LH, HL, HH coefficients
        for i, (title, coeff) in enumerate(
            zip(["LH", "HL", "HH"], yh[0].unbind(dim=2))
        ):
            plt.subplot(2, 2, i + 2)
            plt.imshow(coeff[0, 0].cpu().detach().numpy(), cmap="gray")
            # plt.title(f"{title} Coefficients")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    yl_watermark, yh_watermark = dwt2(watermark)

    display_dwt_coefficients(yl_image, yh_image)
    # Debug prints
    # print(f"Image LL shape: {yl_image.shape}")
    # print(f"Watermark LL shape: {yl_watermark.shape}")
    # print(f"Image highpass shape: {yh_image[0].shape}")
    # print(f"Watermark highpass shape: {yh_watermark[0].shape}")

    # Ensure watermark coefficients match image coefficient dimensions
    if yl_image.shape != yl_watermark.shape:
        yl_watermark = torch.nn.functional.interpolate(
            yl_watermark, size=yl_image.shape[2:], mode="bilinear", align_corners=False
        )

    # We need to ensure yh_image and yh_watermark have compatible shapes
    # These are lists where each element is a tensor of shape (N, C, 3, H, W)
    new_yh_watermark = []
    for i, (img_coeff, wm_coeff) in enumerate(zip(yh_image, yh_watermark)):
        if img_coeff.shape != wm_coeff.shape:
            # Resize the watermark coefficient to match the image coefficient
            # We need to reshape for nn.functional.interpolate
            b, c, three, h, w = img_coeff.shape
            wm_reshaped = wm_coeff.view(b, c * three, h, w)
            img_reshaped = img_coeff.view(b, c * three, h, w)

            wm_resized = torch.nn.functional.interpolate(
                wm_reshaped, size=(h, w), mode="bilinear", align_corners=False
            )

            # Reshape back to original format
            wm_resized = wm_resized.view(b, c, three, h, w)
            new_yh_watermark.append(wm_resized)
        else:
            new_yh_watermark.append(wm_coeff)

    # Apply embedding with alpha factors
    yl_embedded = yl_image + alphas[0] * yl_watermark

    # Apply embedding to highpass coefficients
    yh_embedded = []
    for i, (img_coeff, wm_coeff) in enumerate(zip(yh_image, new_yh_watermark)):
        # i+1 because alphas[0] is used for LL
        alpha_idx = min(i + 1, len(alphas) - 1)
        yh_embedded.append(img_coeff + alphas[alpha_idx] * wm_coeff)

    # Reconstruct the image using IDWT
    watermarked = idwt2((yl_embedded, yh_embedded))

    # Clamp values to valid range
    watermarked = torch.clamp(watermarked, 0, 1)

    return watermarked


def embed_watermark_dct(image: Tensor, watermark: Tensor, alpha: Tensor) -> Tensor:
    """
    Embeds watermark into image using Discrete Cosine Transform (DCT).

    :param image: Image tensor.
    :param watermark: Watermark tensor.
    :param alpha: Tensor scaling factor for watermark.

    :return: Watermarked image tensor.
    """
    # Convert tensor alpha to float if needed
    alpha_float = alpha.item() if isinstance(alpha, Tensor) else alpha

    # Ensure watermark has the same spatial dimensions as the image
    if image.shape[2:] != watermark.shape[2:]:
        watermark = torch.nn.functional.interpolate(
            watermark, size=image.shape[2:], mode="bilinear", align_corners=False
        )

    # Make watermark match image channels if needed
    if image.shape[1] != watermark.shape[1]:
        if image.shape[1] > watermark.shape[1]:
            watermark = watermark.repeat(1, image.shape[1] // watermark.shape[1], 1, 1)

    # Apply DCT to entire tensors
    # Keep tensors on their original device to maintain gradient flow
    image_device = image.device
    watermark_device = watermark.device

    image_dct = dct(image)
    watermark_dct = dct(watermark)

    def display_dct_coefficients(dct_coeffs: Tensor) -> None:
        """
        Display the DCT coefficients using matplotlib.

        :param dct_coeffs: DCT coefficients tensor.
        """
        # Convert to numpy for visualization
        dct_numpy = dct_coeffs[0, 0].cpu().detach().numpy()

        # Display the DCT coefficients
        plt.figure(figsize=(8, 6))
        plt.imshow(dct_numpy, cmap="gray")
        plt.title("DCT Coefficients")
        plt.axis("off")
        plt.colorbar()
        plt.show()

    # Call the function to display DCT coefficients
    display_dct_coefficients(image_dct)

    # Embed watermark
    watermarked_dct = image_dct + alpha_float * watermark_dct

    # Apply IDCT to get the watermarked image
    watermarked = idct(watermarked_dct)  # Fixed: using idct instead of dct for inverse

    # Clamp values to valid range
    watermarked = torch.clamp(watermarked, 0, 1)

    return watermarked
