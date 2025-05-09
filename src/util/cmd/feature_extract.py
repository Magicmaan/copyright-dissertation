import argparse
import os
from typing import Any, List, Tuple, Optional, Dict, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import math
from collections import OrderedDict
import matplotlib.pyplot as plt


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VGG19_CONFIG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]


class VGG(nn.Module):
    """
    VGG19 model for feature extraction.
    """

    def __init__(self) -> None:
        super(VGG, self).__init__()
        self.features = self._make_layers(VGG19_CONFIG)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor after passing through the network
        """
        x = self.features(x)
        return x

    def _make_layers(self, cfg: List) -> nn.Sequential:
        """
        Create the layers of the VGG19 model.

        Args:
            cfg: Configuration list specifying the architecture

        Returns:
            Sequential container of layers
        """
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def extract_features(
        self, x: torch.Tensor
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Extract features from different layers of the model.

        Args:
            x: Input tensor

        Returns:
            Tuple containing:
                - List of numpy arrays for visualization
                - List of tensors for further processing
        """
        features_np = []
        features_tensor = []

        # Get features from each layer
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features_np.append(x.cpu().detach().numpy())
                features_tensor.append(x.clone())

        return features_np, features_tensor


def load_and_preprocess_image(
    image_path: str, target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Load and preprocess an image for the VGG19 model.

    Args:
        image_path: Path to the input image.
        target_size: Target size for resizing the image.

    Returns:
        Preprocessed image tensor ready for feature extraction.
    """
    assert os.path.exists(image_path), f"Image path does not exist: {image_path}"

    # Load image with PIL
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)

    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)

    # Preprocess: channel first and normalize
    img_array = img_array.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add batch dimension

    # Normalize using ImageNet mean and std
    mean = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)
    img_tensor = img_tensor - mean

    return img_tensor


def extract_features(
    image_path: str, model: Optional[VGG] = None
) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
    """
    Extract features from an image using VGG19.

    Args:
        image_path: Path to the input image.
        model: VGG19 model to use. If None, a new model will be created.

    Returns:
        Tuple of feature maps as numpy arrays and tensors.
    """
    if model is None:
        model = VGG().to(DEVICE)
        model.eval()

    # Load and preprocess image
    img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(DEVICE)

    # Extract features
    with torch.no_grad():
        features_np, features_tensor = model.extract_features(img_tensor)

    return features_np, features_tensor


def mse_difference(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Args:
        image1: The first image tensor
        image2: The second image tensor

    Returns:
        The MSE value (lower is better, 0 is perfect match)
    """
    assert image1 is not None and image2 is not None, "Both images must be provided"
    assert isinstance(image1, torch.Tensor) and isinstance(
        image2, torch.Tensor
    ), "Inputs must be torch tensors"
    assert image1.shape == image2.shape, "Images must have the same shape"

    mse = F.mse_loss(image1, image2).item()
    return mse


def psnr(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1: The first image tensor
        image2: The second image tensor

    Returns:
        The PSNR value (higher is better)
    """
    mse = mse_difference(image1, image2)
    if mse == 0:
        return float("inf")  # No noise, perfect match

    max_pixel_value = 255.0
    psnr_value = 20 * math.log10(max_pixel_value / math.sqrt(mse))
    return psnr_value


def ssim(image1: torch.Tensor, image2: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Simplified implementation for standalone use.

    Args:
        image1: The first image tensor
        image2: The second image tensor

    Returns:
        The SSIM value (higher is better, 1 is perfect match)
    """
    # Constants for stability
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Calculate mean, variance and covariance
    mu1 = F.avg_pool2d(image1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(image2, kernel_size=11, stride=1, padding=5)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(image1**2, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(image2**2, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = (
        F.avg_pool2d(image1 * image2, kernel_size=11, stride=1, padding=5) - mu1_mu2
    )

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    return torch.mean(ssim_map).item()


def perceptual_difference(
    image1_features: List[torch.Tensor],
    image2_features: List[torch.Tensor],
    display_features: bool = False,
) -> float:
    """
    Calculate the perceptual difference between two images using their feature maps.

    Args:
        image1_features: List of feature tensors from the first image
        image2_features: List of feature tensors from the second image
        display_features: Whether to display the feature maps (not implemented in standalone)

    Returns:
        The perceptual difference (lower is better, 0 is perfect match)
    """
    assert len(image1_features) == len(
        image2_features
    ), "Feature lists must be the same length"

    # Scale features and calculate differences (following comparison.py)
    difference = []
    for i in range(len(image1_features)):
        features1 = image1_features[i] * 10
        features2 = image2_features[i] * 10

        diff = (features1 - features2) * 100
        difference.append(diff)

    # Flatten and concatenate all features
    image1_flattened = torch.cat([f.flatten() for f in image1_features], dim=0)
    image2_flattened = torch.cat([f.flatten() for f in image2_features], dim=0)

    # Calculate MSE between feature vectors (using the same method as comparison.py)
    mse = torch.mean((image1_flattened - image2_flattened) ** 2).item()
    return mse


def compare_images(
    image1_path: str,
    image2_path: str,
    method: str = "mse",
    display_features: bool = False,
) -> float:
    """
    Compare two images using the specified method.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        method: Comparison method ('mse', 'ssim', 'psnr', 'perceptual')
        display_features: Whether to display feature maps (only for perceptual method)

    Returns:
        Comparison result as a float
    """
    # Load VGG19 model
    model = VGG().to(DEVICE)
    model.eval()

    # Load and preprocess images
    img1_tensor = load_and_preprocess_image(image1_path)
    img2_tensor = load_and_preprocess_image(image2_path)

    if method.lower() == "mse":
        return mse_difference(img1_tensor, img2_tensor)
    elif method.lower() == "ssim":
        return ssim(img1_tensor, img2_tensor)
    elif method.lower() == "psnr":
        return psnr(img1_tensor, img2_tensor)
    elif method.lower() == "perceptual":
        # Extract features
        img1_tensor_device = img1_tensor.to(DEVICE)
        img2_tensor_device = img2_tensor.to(DEVICE)

        with torch.no_grad():
            _, img1_features = model.extract_features(img1_tensor_device)
            _, img2_features = model.extract_features(img2_tensor_device)

        return perceptual_difference(img1_features, img2_features, display_features)
    else:
        raise ValueError(f"Unknown comparison method: {method}")


def save_feature_maps(
    feature_maps: List[np.ndarray],
    output_dir: str,
    filename_prefix: str,
    conv_layers: Optional[List[int]] = None,
) -> None:
    """
    Save feature maps as images using matplotlib.

    Args:
        feature_maps: List of numpy arrays containing feature maps
        output_dir: Directory to save the images
        filename_prefix: Prefix for the saved files
        conv_layers: List of layer indices to save (if None, save all)
    """
    assert (
        feature_maps is not None and len(feature_maps) > 0
    ), "Feature maps must be provided"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use all layers if not specified
    if conv_layers is None:
        conv_layers = list(range(len(feature_maps)))

    # Loop through feature maps and save each one
    for i, layer_idx in enumerate(conv_layers):
        if layer_idx >= len(feature_maps):
            continue

        feature_map = feature_maps[layer_idx]

        # For each feature map (which may have multiple channels), save up to 16 channels
        batch_size, num_channels = feature_map.shape[0], feature_map.shape[1]

        # Determine grid size (maximum 4x4 grid)
        grid_size = min(4, int(np.ceil(np.sqrt(min(num_channels, 16)))))

        # Create figure for saving
        plt.figure(figsize=(20, 20))
        plt.suptitle(
            f"Feature Map {layer_idx+1} - {feature_map.shape[1]} channels", fontsize=16
        )

        # Plot up to 16 channels of the feature map
        for j in range(min(grid_size * grid_size, num_channels)):
            plt.subplot(grid_size, grid_size, j + 1)
            plt.imshow(feature_map[0, j], cmap="viridis")
            plt.title(f"Channel {j+1}")
            plt.axis("off")

        # Save the figure
        output_path = os.path.join(
            output_dir, f"{filename_prefix}_layer_{layer_idx+1}.png"
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

    print(f"Saved {len(conv_layers)} feature maps to {output_dir}")


def display_feature_maps(
    feature_maps: List[np.ndarray],
    conv_layers: Optional[List[int]] = None,
    title: str = "Feature Maps",
) -> None:
    """
    Display feature maps using matplotlib.

    Args:
        feature_maps: List of numpy arrays containing feature maps
        conv_layers: List of layer indices to display (if None, display all)
        title: Title for the figure
    """
    assert (
        feature_maps is not None and len(feature_maps) > 0
    ), "Feature maps must be provided"

    # Use all layers if not specified
    if conv_layers is None:
        conv_layers = list(range(len(feature_maps)))

    # Loop through feature maps and display each one
    for i, layer_idx in enumerate(conv_layers):
        if layer_idx >= len(feature_maps):
            continue

        feature_map = feature_maps[layer_idx]

        # For each feature map (which may have multiple channels), display up to 16 channels
        batch_size, num_channels = feature_map.shape[0], feature_map.shape[1]

        # Determine grid size (maximum 4x4 grid)
        grid_size = min(4, int(np.ceil(np.sqrt(min(num_channels, 16)))))

        # Create figure for display
        plt.figure(figsize=(20, 20))
        plt.suptitle(
            f"{title} - Layer {layer_idx+1} - {feature_map.shape[1]} channels",
            fontsize=16,
        )

        # Plot up to 16 channels of the feature map
        for j in range(min(grid_size * grid_size, num_channels)):
            plt.subplot(grid_size, grid_size, j + 1)
            plt.imshow(feature_map[0, j], cmap="viridis")
            plt.title(f"Channel {j+1}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def main() -> None:
    """
    Main function to parse arguments and extract features or compare images.
    """
    parser = argparse.ArgumentParser(
        description="Extract features from images using VGG19 or compare two images."
    )
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "--compare", "-c", type=str, help="Path to a second image for comparison."
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default="mse",
        choices=["mse", "ssim", "psnr", "perceptual"],
        help="Comparison method when comparing two images.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./feature_maps",
        help="Directory to save feature maps (if --save-maps is specified)",
    )
    parser.add_argument(
        "--save-maps", "-s", action="store_true", help="Save feature maps as images"
    )
    parser.add_argument(
        "--display",
        "-d",
        action="store_true",
        help="Display feature maps (may not work in all environments)",
    )
    args = parser.parse_args()

    # Create VGG19 model
    model = VGG().to(DEVICE)
    model.eval()

    if args.compare:
        # Compare two images
        comparison_result = compare_images(
            args.image_path, args.compare, args.method, args.display
        )

        if args.method.lower() == "mse" or args.method.lower() == "perceptual":
            print(
                f"{args.method.upper()} difference: {comparison_result} (lower is better)"
            )
        elif args.method.lower() == "ssim":
            print(f"SSIM: {comparison_result} (higher is better, 1 is perfect match)")
        elif args.method.lower() == "psnr":
            print(f"PSNR: {comparison_result} dB (higher is better)")

        # If saving feature maps is requested and we're using perceptual difference
        if args.save_maps and args.method.lower() == "perceptual":
            # Extract features for both images
            features_np1, _ = extract_features(args.image_path, model)
            features_np2, _ = extract_features(args.compare, model)

            # Save feature maps
            image1_name = os.path.splitext(os.path.basename(args.image_path))[0]
            image2_name = os.path.splitext(os.path.basename(args.compare))[0]

            # Calculate difference maps
            diff_maps = []
            for i in range(len(features_np1)):
                # Scale like in perceptual_difference
                features1 = features_np1[i] * 10
                features2 = features_np2[i] * 10
                diff_map = (features1 - features2) * 100
                diff_maps.append(diff_map)

            # Save maps
            conv_layers = list(range(len(features_np1)))
            save_feature_maps(
                features_np1, args.output_dir, f"{image1_name}", conv_layers
            )
            save_feature_maps(
                features_np2, args.output_dir, f"{image2_name}", conv_layers
            )
            save_feature_maps(
                diff_maps,
                args.output_dir,
                f"{image1_name}_vs_{image2_name}_diff",
                conv_layers,
            )
    else:
        # Extract features from a single image
        features_np, features_tensor = extract_features(args.image_path, model)
        print(f"Extracted features from {len(features_tensor)} layers")
        print(f"First feature map shape: {features_tensor[0].shape}")

        # Display feature maps if requested
        if args.display:
            conv_layers = list(range(len(features_np)))
            display_feature_maps(features_np, conv_layers, "Feature Maps")

        # Save feature maps if requested
        if args.save_maps:
            image_name = os.path.splitext(os.path.basename(args.image_path))[0]
            conv_layers = list(range(len(features_np)))
            save_feature_maps(features_np, args.output_dir, image_name, conv_layers)


if __name__ == "__main__":
    main()
