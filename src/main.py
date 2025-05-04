import random
from typing import Literal
from PIL import Image
import PIL
import PIL.Image
import PIL.ImageFile
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from GAN import *
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
from GAN import adversarialLoss
from GAN import Discriminator as DiscriminatorExt
from GAN import discriminatorLoss
from GAN import Generator
from GAN import PerceptualLoss
from util.NST import performNST
from util.comparison import *
from util.dctdwt import (
    embedWatermark,
    extract_watermark_dwt,
    extract_watermark_dct,
)
import torch
from util.debug import display_image_tensors
from util.image import imageToTensor, preprocessImage, tensorToImage
import os

import tkinter as tk
from tkinter import Canvas
from PIL import ImageTk, Image
import time

# Load VGG19 model
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")
# Use GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)  # for reproducibility
torch.use_deterministic_algorithms(True)  # for reproducibility

# https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# load assets
DATA_PATH = Path("data")

TRAINING_PATH = DATA_PATH / "training"

# load watermark
WATERMARK: Image = Image.open(DATA_PATH / "watermark.jpg").convert("RGB")
assert WATERMARK is not None, "Watermark not found."


# load content and style images
CONTENT_IMAGES_LIST: list[Path] = [
    Image.open(image) for image in list(DATA_PATH.glob("content/*.jpg"))
]
STYLE_IMAGES_LIST: list[Path] = [
    Image.open(image) for image in list(DATA_PATH.glob("style/*.jpg"))
]
assert len(CONTENT_IMAGES_LIST) > 0, "No content images found."
assert len(STYLE_IMAGES_LIST) > 0, "No style images found."


# hyper parameters for training
EPOCHS = 100


class WatermarkDetector(nn.Module):
    def __init__(self):
        super(WatermarkDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, content, watermarked):
        x = torch.cat([content, watermarked], dim=1)  # Concatenate along channel dim
        return self.model(x)


class ImageDataset(Dataset):
    def __init__(self, images_path: Path):
        self.images_path = images_path
        self.image_files = [f for f in images_path.glob("*.jpg") if f.is_file()]
        assert self.image_files, "No .jpg files found in the specified path."

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        image_path = self.image_files[index]
        assert image_path.exists(), f"File {image_path} does not exist."
        image = preprocessImage(Image.open(image_path).convert("RGB"), DEVICE).to(
            DEVICE
        )
        print(f"Loaded image: {image_path}")
        return image


def plot_results(
    epochs: int,
    total_loss: list[float],
    pixel_loss: list[float],
    watermark_loss: list[float],
    adversarial_loss: list[float],
    dct_alpha: list[float],
    dwt_alpha: list[list[float, float, float, float]],
    save_path: Path = None,
    show_figure: bool = False,
):
    """Plot the training progress of the watermarking system."""
    try:

        plt.ion()  # Turn on interactive mode
        plt.figure(figsize=(12, 8))

        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(range(epochs), total_loss, label="Total Loss", color="red")
        plt.plot(range(epochs), pixel_loss, label="Pixel Loss", color="blue")
        plt.plot(range(epochs), watermark_loss, label="Watermark Loss", color="green")
        plt.plot(
            range(epochs), adversarial_loss, label="Adversarial Loss", color="purple"
        )
        plt.yscale("linear")  # Use linear scale for loss
        plt.title("Loss over epochs (Log Scale)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Log Scale)")
        plt.legend()

        # Plot alphas
        plt.subplot(2, 1, 2)
        plt.plot(range(epochs), dct_alpha, label="DCT Alpha", color="blue")

        # Define colors for DWT alphas
        dwt_colours = ["orange", "green", "purple", "brown"]

        # Plot each DWT alpha separately with its own color
        for i in range(4):
            plt.plot(
                range(epochs),
                [alpha[i] for alpha in dwt_alpha],
                alpha=1 / (i + 1),
                color=dwt_colours[i],
                linestyle="dotted",
                linewidth=2,
                label=f"DWT Alpha {i+1}",
            )

        plt.yscale("log")  # Use log scale for alphas
        plt.title("Alpha Values over epochs (Log Scale)")
        plt.xlabel("Epoch")
        plt.ylabel("Alpha Value (Log Scale)")
        plt.legend()

        plt.tight_layout()

        plt.draw()
        plt.pause(0.001)

        if show_figure:
            plt.show()

        if save_path:
            plt.savefig(save_path / "training_progress_log_scale.png")

        plt.close()
    except Exception as e:
        print(f"Error plotting results: {e}")


def train(
    content_tensor: Tensor,
    watermark_tensor: Tensor,
    style_tensor: Tensor,
    epochs=50,
    show_figure=False,
    save_path: Path = None,
):
    """
    Train the watermarking system to find optimal alpha values.

    Args:
        content_tensor: The original content image
        watermark_tensor: The watermark to embed
        style_tensor: The style image (not directly used in training)
        epochs: Number of training epochs

    Returns:
        Tuple of optimized alpha parameters (dwt_alphas, dct_alpha)
    """

    # Initialize alpha parameters with requires_grad=True - using smaller initial values
    dwt_a1 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a2 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a3 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a4 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dct_alpha = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))

    # initialise adversary
    adversary = WatermarkDetector().to(DEVICE)
    adversary_optimiser = Adam(adversary.parameters(), lr=0.005)

    # Create optimizer with appropriate learning rate
    optimiser = Adam([dwt_a1, dwt_a2, dwt_a3, dwt_a4, dct_alpha], lr=0.01)

    # Loss weights - giving more weight to the watermark loss
    parameters = {
        "pixel": 0.8,  # Visual quality preservation
        "adversarial": 0.8,  # Adversarial loss for watermark detection
        "dwt_weights": 0.9,  # DWT alphas weight
        "watermark": 1.5,  # Watermark quality preservation
    }

    # Arrays to track loss and alpha values
    history = {
        "total_loss": [],
        "pixel_loss": [],
        "watermark_loss": [],
        "adversarial_loss": [],
        "dct_alpha": [],
        "dwt_alpha": [],
    }

    stall_count = 0
    stall_count_max = 2

    base_styled = performNST(content_tensor, style_tensor, iterations=5, mode="adain")
    base_styled = base_styled.to(DEVICE)

    print("Beginning training...")
    for epoch in range(epochs):
        # Reset gradients
        optimiser.zero_grad()
        adversary_optimiser.zero_grad()

        # Forward pass - embed watermark using current alpha values
        watermarked, extracted, _, _, _, _ = embedWatermark(
            content_tensor,
            watermark_tensor,
            [dwt_a1, dwt_a2, dwt_a3, dwt_a4],
            dct_alpha,
            display=False,
        )

        styled = performNST(
            watermarked,
            style_tensor,  # Use the style image for NST
            iterations=5,
            mode="adain",
        )
        styled = styled.to(DEVICE)

        styled_extracted = extract_watermark_dct(
            content_tensor, styled, alpha=dct_alpha
        )

        # get adversary prediction for if watermark exists
        prediction = adversary(content_tensor, styled)
        label = torch.ones_like(prediction, device=DEVICE)
        adversary_loss = torch.nn.functional.binary_cross_entropy(prediction, label)

        # Calculate losses
        pixel_loss = perceptualDifference(base_styled, styled)
        watermark_loss = pixel_difference_float(watermark_tensor, styled_extracted)

        # Total loss
        total_loss = (
            parameters["pixel"] * pixel_loss
            + parameters["watermark"] * watermark_loss
            + parameters["adversarial"] * adversary_loss
        )

        # Regularisation loss for DWT alphas
        def relative_decay_loss(alphas: list[torch.Tensor]) -> torch.Tensor:
            penalty = 0.0
            for i in range(1, len(alphas)):
                target_alpha = alphas[i - 1] * parameters["dwt_weights"]
                penalty += alphas[i] - target_alpha
            return penalty

        reg_loss = relative_decay_loss([dwt_a1, dwt_a2, dwt_a3, dwt_a4])
        total_loss += reg_loss

        # Backpropagate
        total_loss.backward()
        optimiser.step()
        adversary_optimiser.step()

        # Clamp alphas to positive values
        with torch.no_grad():
            dwt_a1.data.clamp_(min=0.00001, max=0.1)
            dwt_a2.data.clamp_(min=0.00001, max=0.1)
            dwt_a3.data.clamp_(min=0.00001, max=0.1)
            dwt_a4.data.clamp_(min=0.00001, max=0.1)
            dct_alpha.data.clamp_(min=0.00001, max=0.1)

        # Check for loss stalling
        if history["total_loss"] and len(history["total_loss"]) > 1:
            if total_loss.item() == history["total_loss"][-1]:
                stall_count += 1
                if stall_count >= stall_count_max:
                    print(
                        f"Training stalled for {stall_count_max} epochs. Stopping early."
                    )
                    epochs = epoch
                    break
            else:
                stall_count = 0

        # Store history
        history["total_loss"].append(total_loss.item())
        history["pixel_loss"].append(pixel_loss.item())
        history["watermark_loss"].append(watermark_loss.item())
        history["adversarial_loss"].append(adversary_loss.item())
        history["dct_alpha"].append(dct_alpha.item())
        history["dwt_alpha"].append(
            [dwt_a1.item(), dwt_a2.item(), dwt_a3.item(), dwt_a4.item()]
        )

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss.item():.6f}, "
                f"Pixel Loss: {pixel_loss.item():.6f}, "
                f"Watermark Loss: {watermark_loss.item():.6f}, "
                f"Adversarial Loss: {adversary_loss.item():.6f}"
            )

    # Plot the training progress
    if epochs > 0:
        plot_results(
            epochs=epochs,
            total_loss=history["total_loss"],
            pixel_loss=history["pixel_loss"],
            watermark_loss=history["watermark_loss"],
            adversarial_loss=history["adversarial_loss"],
            dct_alpha=history["dct_alpha"],
            dwt_alpha=history["dwt_alpha"],
            save_path=save_path,
            show_figure=show_figure,
        )

    print("Training complete.")
    print(f"Final Total Loss: {history['total_loss'][-1]:.6f}")
    print(f"Final DCT Alpha: {dct_alpha.item()}")
    print(
        f"Final DWT Alphas: {dwt_a1.item()}, {dwt_a2.item()}, {dwt_a3.item()}, {dwt_a4.item()}"
    )

    return (
        [
            dwt_a1.detach().item(),
            dwt_a2.detach().item(),
            dwt_a3.detach().item(),
            dwt_a4.detach().item(),
        ],
        dct_alpha.item(),
        watermarked,
        extracted,
        styled,
        styled_extracted,
        base_styled,
    )


def batch_watermark_size(
    style_image: Path = None,
    content_image: Path = None,
    watermarks: list[Path] = None,
    epochs: int = 25,
    ext_path: Path = Path(""),
):
    assert style_image.exists(), f"Style image {style_image} does not exist."
    assert content_image.exists(), f"Content image {content_image} does not exist."
    assert watermarks, "Watermark list is empty."
    assert len(watermarks) > 1, "Watermark list is 1 or under."

    print("Performing watermark variable size analysis...")

    watermark_tensors = [
        preprocessImage(Image.open(watermark).convert("RGB"), DEVICE).to(DEVICE)
        for watermark in watermarks
    ]

    content_tensor = preprocessImage(
        Image.open(content_image).convert("RGB"), DEVICE
    ).to(DEVICE)
    style_tensor = preprocessImage(Image.open(style_image).convert("RGB"), DEVICE).to(
        DEVICE
    )

    for i, watermark in enumerate(watermark_tensors):
        print(f"Processing watermark {i + 1}/{len(watermarks)}...")

        # Create a new directory for each watermark
        output_path = (
            TRAINING_PATH / "output" / ext_path / "watermark_size" / str(i + 1)
        )
        output_path.mkdir(parents=True, exist_ok=True)
        assert output_path.exists(), f"Output path {output_path} does not exist."

        (
            dwt_alphas,
            dct_alpha,
            watermarked_image,
            extracted_watermark,
            styled_watermarked_image,
            styled_extracted_watermark,
            styled,
        ) = train(
            content_tensor=content_tensor,
            watermark_tensor=watermark,
            style_tensor=style_tensor,
            epochs=epochs,
            save_path=output_path,
        )

        # Save the content and style tensors as images
        content_image_path = output_path / f"content.png"
        content_image_pil = tensorToImage(content_tensor)
        content_image_pil.save(content_image_path)

        style_image_path = output_path / f"style.png"
        style_image_pil = tensorToImage(style_tensor)
        style_image_pil.save(style_image_path)

        print(f"Saved content image to {content_image_path}")
        print(f"Saved style image to {style_image_path}")

        # Save the watermarked image
        watermarked_image_path = output_path / f"watermarked.png"
        watermarked_image_pil = tensorToImage(watermarked_image)
        watermarked_image_pil.save(watermarked_image_path)

        styled_watermarked_image_path = output_path / f"styled_watermarked.png"
        styled_watermarked_image_pil = tensorToImage(styled_watermarked_image)
        styled_watermarked_image_pil.save(styled_watermarked_image_path)

        styled_no_watermark_path = output_path / f"styled_no_watermark.png"
        styled_no_watermark_pil = tensorToImage(styled)
        styled_no_watermark_pil.save(styled_no_watermark_path)

        styled_extracted_watermark_path = (
            output_path / f"styled_extracted_watermark.png"
        )
        styled_extracted_watermark_pil = tensorToImage(styled_extracted_watermark)
        styled_extracted_watermark_pil.save(styled_extracted_watermark_path)

        extracted_watermark_path = output_path / f"extracted_watermark.png"
        extracted_watermark_pil = tensorToImage(extracted_watermark)
        extracted_watermark_pil.save(extracted_watermark_path)

        # Calculate the difference between styled extracted and extracted watermarks
        extracted_comparison = torch.abs(
            styled_extracted_watermark - extracted_watermark
        )
        # Save the extracted comparison image
        extracted_comparison_path = output_path / "extracted_comparison.png"
        extracted_comparison_pil = tensorToImage(extracted_comparison)
        extracted_comparison_pil.save(extracted_comparison_path)

        # Calculate the difference between styled and styled watermarked images
        styled_comparison = torch.abs(styled - styled_watermarked_image)

        # Save the styled comparison image
        styled_comparison_path = output_path / "styled_comparison.png"
        styled_comparison_pil = tensorToImage(styled_comparison)
        styled_comparison_pil.save(styled_comparison_path)

        # Save DWT alphas and DCT alpha to a file
        alphas_file_path = output_path / "alphas.txt"
        with open(alphas_file_path, "w") as f:
            f.write(f"DWT Alphas: {dwt_alphas}\n")
            f.write(f"DCT Alpha: {dct_alpha}\n")
        print(f"Saved alpha values to {alphas_file_path}")

        # Create a matplotlib figure
        def plot_images(
            content_image,
            style_image,
            watermark_image,
            watermarked_image,
            styled_watermarked_image,
            extracted_watermark,
            save_path,
        ):
            plt.figure(figsize=(15, 10))

            # First row
            plt.subplot(2, 3, 1)
            plt.imshow(tensorToImage(content_image).convert("RGB"))
            plt.title("Content Image")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(tensorToImage(style_image).convert("RGB"))
            plt.title("Style Image")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(tensorToImage(watermark_image).convert("RGB"))
            plt.title("Watermark")
            plt.axis("off")

            # Second row
            plt.subplot(2, 3, 4)
            plt.imshow(tensorToImage(watermarked_image).convert("RGB"))
            plt.title("Watermarked Image")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(tensorToImage(styled_watermarked_image).convert("RGB"))
            plt.title("Styled Watermarked Image")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(tensorToImage(extracted_watermark).convert("RGB"))
            plt.title("Extracted Watermark")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path / "figure.png")
            plt.close()

            # print(f"Saved styled comparison image to {styled_comparison_path}")

        # Call the function to plot the images
        plot_images(
            content_image=content_tensor,
            style_image=style_tensor,
            watermark_image=watermark,
            watermarked_image=watermarked_image,
            styled_watermarked_image=styled_watermarked_image,
            extracted_watermark=styled_extracted_watermark,
            save_path=output_path,
        )

        # get metrics and save to analysis.txt
        (
            pixel_diff_styled,
            pixel_diff_extracted,
            pixel_diff_styled_vs_watermarked,
            psnr_styled,
            psnr_extracted,
            psnr_styled_vs_watermarked,
            perceptual_diff_styled,
            perceptual_diff_extracted,
            perceptual_diff_styled_vs_watermarked,
            structural_diff_styled,
            structural_diff_extracted,
            structural_diff_styled_vs_watermarked,
        ) = analyse(
            original_image=content_tensor,
            watermarked_image=watermarked_image,
            extracted_watermark=extracted_watermark,
            styled_watermarked_image=styled_watermarked_image,
            styled_extracted_watermark=styled_extracted_watermark,
            styled=styled,
        )
        analysis_file_path = output_path / "analysis.txt"
        with open(analysis_file_path, "w") as f:
            f.write(
                f"Pixel Difference (Watermarked vs Styled Watermarked): {pixel_diff_styled}\n"
            )
            f.write(f"PSNR (Watermarked vs Styled Watermarked): {psnr_styled}\n")
            f.write(
                f"Perceptual Difference (Watermarked vs Styled Watermarked): {perceptual_diff_styled}\n"
            )
            f.write(
                f"Structural Difference (Watermarked vs Styled Watermarked): {structural_diff_styled}\n"
            )
            f.write("\n")
            f.write(
                f"Pixel Difference (Extracted vs Styled Extracted): {pixel_diff_extracted}\n"
            )
            f.write(f"PSNR (Extracted vs Styled Extracted): {psnr_extracted}\n")
            f.write(
                f"Perceptual Difference (Extracted vs Styled Extracted): {perceptual_diff_extracted}\n"
            )
            f.write(
                f"Structural Difference (Extracted vs Styled Extracted): {structural_diff_extracted}\n"
            )
            f.write("\n")
            f.write(
                f"Pixel Difference (Styled vs Styled Watermarked): {pixel_diff_styled_vs_watermarked}\n"
            )
            f.write(
                f"PSNR (Styled vs Styled Watermarked): {psnr_styled_vs_watermarked}\n"
            )
            f.write(
                f"Perceptual Difference (Styled vs Styled Watermarked): {perceptual_diff_styled_vs_watermarked}\n"
            )
            f.write(
                f"Structural Difference (Styled vs Styled Watermarked): {structural_diff_styled_vs_watermarked}\n"
            )
        print(f"Saved analysis results to {analysis_file_path}")

        if batch_type == "watermark_resize":
            current_watermark += 1


batch_type = Literal[
    "1 to 1", "random", "same style", "same content", "watermark_resize"
]


def batch_watermark(
    batch_type: batch_type = "random",
    style_image: Path = None,
    content_image: Path = None,
    watermark: list[Path] = None,
    epochs: int = 25,
    count: int = 0,
    ext_path: Path = Path(""),
):
    """
    Batch process the watermarking system to embed watermarks into multiple images.

    Args:
        batch_type: Type of batch processing to perform. Options are:
            - "1 to 1": Match content and style images 1 to 1
            - "random": Randomly match content and style images
            - "same style": Match content images with the same style image
            - "same content": Match style images with the same content image
        style_image: Path to the style image (if using same style or same content)
        content_image: Path to the content image (if using same style or same content)

    Returns:
        List of tuples containing the watermarked image and extracted watermark
    """

    content_path = TRAINING_PATH / "content"
    style_path = TRAINING_PATH / "style"

    print("Loading content and style images...")
    print(f"Content path: {content_path}")
    print(f"Style path: {style_path}")
    # load content and style images

    current_watermark = 0
    if watermark and len(watermark) > 0:
        assert isinstance(watermark, (list))("Watermark must be a list of images.")

        images = [
            Image.open(watermark_image).convert("RGB") for watermark_image in watermark
        ]
        # load tensors
        watermark_tensor = [
            preprocessImage(watermark_image, DEVICE).to(DEVICE)
            for watermark_image in images
        ]

        assert watermark is not None, "Watermark not found."
    else:
        watermark_tensor = [preprocessImage(WATERMARK, DEVICE).to(DEVICE)]

    content_dataset = ImageDataset(content_path)
    style_dataset = ImageDataset(style_path)

    content_dataloader: DataLoader = None
    style_dataloader: DataLoader = None

    print("Running batch watermarking...")
    print(f"Batch type: {batch_type}")
    match batch_type:
        case "1 to 1":
            # 1 to 1 matching of content and style images
            # content_1 + style_1, ...
            content_dataloader = DataLoader(
                content_dataset, batch_size=1, shuffle=False
            )
            style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=False)
        case "random":
            # Randomly match content and style images
            # content_1 + style_3, content_2 + style_1, ...
            content_dataloader = DataLoader(content_dataset, batch_size=1, shuffle=True)
            style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=True)
        case "same style":
            # Match content images with the same style image
            # content_1 + style_1, content_2 + style_1, ...
            content_dataloader = DataLoader(
                content_dataset, batch_size=1, shuffle=False
            )
            style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=False)
            style_tensor = next(iter(style_dataloader))[0]

            if style_image:
                # If a specific style image is provided, use it
                style_tensor = preprocessImage(
                    Image.open(style_image).convert("RGB"), DEVICE
                ).to(DEVICE)

            style_dataloader = [style_tensor] * len(content_dataset)
        case "same content":
            # Match style images with the same content image
            # content_1 + style_1, content_1 + style_2, ...
            content_dataloader = DataLoader(
                content_dataset, batch_size=1, shuffle=False
            )
            style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=False)

            # Get the first content image and replicate it
            content_tensor = next(iter(content_dataloader))

            if content_image:
                # If a specific content image is provided, use it
                c = preprocessImage(
                    Image.open(content_image).convert("RGB"), DEVICE
                ).to(DEVICE)
                content_tensor[0] = c.unsqueeze(0)
            # Create a list where each item is the same content tensor
            content_dataloader = [content_tensor for _ in range(len(style_dataset))]

        case "watermark_resize":
            content_dataloader = DataLoader(
                content_dataset, batch_size=1, shuffle=False
            )
            style_dataloader = DataLoader(style_dataset, batch_size=1, shuffle=False)

            content_tensor = next(iter(content_dataloader))
            style_tensor = next(iter(style_dataloader))

            if content_image:
                # If a specific content image is provided, use it
                c = preprocessImage(
                    Image.open(content_image).convert("RGB"), DEVICE
                ).to(DEVICE)
                content_tensor[0] = c.unsqueeze(0)
                content_dataloader = [content_tensor for _ in range(len(style_dataset))]

            if style_image:
                # If a specific style image is provided, use it
                s = preprocessImage(Image.open(style_image).convert("RGB"), DEVICE).to(
                    DEVICE
                )
                style_tensor[0] = s.unsqueeze(0)

        case _:
            raise ValueError(
                "Invalid batch type. Choose from '1 to 1', 'random', 'same style', or 'same content' or 'watermark_resize'."
            )

    # Randomise the order of dataloaders
    if isinstance(content_dataloader, DataLoader):
        content_dataloader = list(content_dataloader)
        random.shuffle(content_dataloader)

    if isinstance(style_dataloader, DataLoader):
        style_dataloader = list(style_dataloader)
        random.shuffle(style_dataloader)

    batch_type_output = batch_type.replace(" ", "_")
    results = []
    total_images = min(len(content_dataset), len(style_dataset))
    print(f"Starting batch watermarking for {total_images} images...")

    for i, (content_image, style_image) in enumerate(
        zip(content_dataloader, style_dataloader)
    ):
        if i >= total_images:
            break
        if i >= count:
            break

        start_time = time.time()  # Start timer for the current iteration

        con_path: str = content_dataset.image_files[i].stem
        print(f"Processing image pair {i + 1}/{total_images}...")
        print(f"Content image path: {con_path}")

        sty_path: str = style_dataset.image_files[i].stem
        print(f"Style image path: {sty_path}")

        output_path = TRAINING_PATH / "output" / ext_path / batch_type_output / con_path
        if output_path.exists():
            print(f"Output path {output_path} already exists. Skipping...")
            continue
        output_path.mkdir(parents=True, exist_ok=True)
        content_tensor = content_image[0]
        style_tensor = style_image[0]

        (
            dwt_alphas,
            dct_alpha,
            watermarked_image,
            extracted_watermark,
            styled_watermarked_image,
            styled_extracted_watermark,
            styled,
        ) = train(
            content_tensor,
            watermark_tensor[current_watermark],
            style_tensor,
            epochs=epochs,
            save_path=output_path,
        )

        # Save the content and style tensors as images
        content_image_path = output_path / f"content.png"
        content_image_pil = tensorToImage(content_tensor)
        content_image_pil.save(content_image_path)

        style_image_path = output_path / f"style.png"
        style_image_pil = tensorToImage(style_tensor)
        style_image_pil.save(style_image_path)

        print(f"Saved content image to {content_image_path}")
        print(f"Saved style image to {style_image_path}")

        # Save the watermarked image
        watermarked_image_path = output_path / f"watermarked.png"
        watermarked_image_pil = tensorToImage(watermarked_image)
        watermarked_image_pil.save(watermarked_image_path)

        styled_watermarked_image_path = output_path / f"styled_watermarked.png"
        styled_watermarked_image_pil = tensorToImage(styled_watermarked_image)
        styled_watermarked_image_pil.save(styled_watermarked_image_path)

        styled_no_watermark_path = output_path / f"styled_no_watermark.png"
        styled_no_watermark_pil = tensorToImage(styled)
        styled_no_watermark_pil.save(styled_no_watermark_path)

        styled_extracted_watermark_path = (
            output_path / f"styled_extracted_watermark.png"
        )
        styled_extracted_watermark_pil = tensorToImage(styled_extracted_watermark)
        styled_extracted_watermark_pil.save(styled_extracted_watermark_path)

        extracted_watermark_path = output_path / f"extracted_watermark.png"
        extracted_watermark_pil = tensorToImage(extracted_watermark)
        extracted_watermark_pil.save(extracted_watermark_path)

        # Calculate the difference between styled extracted and extracted watermarks
        extracted_comparison = torch.abs(
            styled_extracted_watermark - extracted_watermark
        )
        # Save the extracted comparison image
        extracted_comparison_path = output_path / "extracted_comparison.png"
        extracted_comparison_pil = tensorToImage(extracted_comparison)
        extracted_comparison_pil.save(extracted_comparison_path)

        # Calculate the difference between styled and styled watermarked images
        styled_comparison = torch.abs(styled - styled_watermarked_image)

        # Save the styled comparison image
        styled_comparison_path = output_path / "styled_comparison.png"
        styled_comparison_pil = tensorToImage(styled_comparison)
        styled_comparison_pil.save(styled_comparison_path)

        # Save DWT alphas and DCT alpha to a file
        alphas_file_path = output_path / "alphas.txt"
        with open(alphas_file_path, "w") as f:
            f.write(f"DWT Alphas: {dwt_alphas}\n")
            f.write(f"DCT Alpha: {dct_alpha}\n")
        print(f"Saved alpha values to {alphas_file_path}")

        # Create a matplotlib figure
        def plot_images(
            content_image,
            style_image,
            watermark_image,
            watermarked_image,
            styled_watermarked_image,
            extracted_watermark,
            save_path,
        ):
            plt.figure(figsize=(15, 10))

            # First row
            plt.subplot(2, 3, 1)
            plt.imshow(tensorToImage(content_image).convert("RGB"))
            plt.title("Content Image")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(tensorToImage(style_image).convert("RGB"))
            plt.title("Style Image")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(tensorToImage(watermark_image).convert("RGB"))
            plt.title("Watermark")
            plt.axis("off")

            # Second row
            plt.subplot(2, 3, 4)
            plt.imshow(tensorToImage(watermarked_image).convert("RGB"))
            plt.title("Watermarked Image")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(tensorToImage(styled_watermarked_image).convert("RGB"))
            plt.title("Styled Watermarked Image")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(tensorToImage(extracted_watermark).convert("RGB"))
            plt.title("Extracted Watermark")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path / "figure.png")
            plt.close()

            # print(f"Saved styled comparison image to {styled_comparison_path}")

        # Call the function to plot the images
        plot_images(
            content_image=content_tensor,
            style_image=style_tensor,
            watermark_image=watermark_tensor,
            watermarked_image=watermarked_image,
            styled_watermarked_image=styled_watermarked_image,
            extracted_watermark=styled_extracted_watermark,
            save_path=output_path,
        )

        # get metrics and save to analysis.txt
        (
            pixel_diff_styled,
            pixel_diff_extracted,
            pixel_diff_styled_vs_watermarked,
            psnr_styled,
            psnr_extracted,
            psnr_styled_vs_watermarked,
            perceptual_diff_styled,
            perceptual_diff_extracted,
            perceptual_diff_styled_vs_watermarked,
            structural_diff_styled,
            structural_diff_extracted,
            structural_diff_styled_vs_watermarked,
        ) = analyse(
            original_image=content_tensor,
            watermarked_image=watermarked_image,
            extracted_watermark=extracted_watermark,
            styled_watermarked_image=styled_watermarked_image,
            styled_extracted_watermark=styled_extracted_watermark,
            styled=styled,
        )
        analysis_file_path = output_path / "analysis.txt"
        with open(analysis_file_path, "w") as f:
            f.write(
                f"Pixel Difference (Watermarked vs Styled Watermarked): {pixel_diff_styled}\n"
            )
            f.write(f"PSNR (Watermarked vs Styled Watermarked): {psnr_styled}\n")
            f.write(
                f"Perceptual Difference (Watermarked vs Styled Watermarked): {perceptual_diff_styled}\n"
            )
            f.write(
                f"Structural Difference (Watermarked vs Styled Watermarked): {structural_diff_styled}\n"
            )
            f.write("\n")
            f.write(
                f"Pixel Difference (Extracted vs Styled Extracted): {pixel_diff_extracted}\n"
            )
            f.write(f"PSNR (Extracted vs Styled Extracted): {psnr_extracted}\n")
            f.write(
                f"Perceptual Difference (Extracted vs Styled Extracted): {perceptual_diff_extracted}\n"
            )
            f.write(
                f"Structural Difference (Extracted vs Styled Extracted): {structural_diff_extracted}\n"
            )
            f.write("\n")
            f.write(
                f"Pixel Difference (Styled vs Styled Watermarked): {pixel_diff_styled_vs_watermarked}\n"
            )
            f.write(
                f"PSNR (Styled vs Styled Watermarked): {psnr_styled_vs_watermarked}\n"
            )
            f.write(
                f"Perceptual Difference (Styled vs Styled Watermarked): {perceptual_diff_styled_vs_watermarked}\n"
            )
            f.write(
                f"Structural Difference (Styled vs Styled Watermarked): {structural_diff_styled_vs_watermarked}\n"
            )
        print(f"Saved analysis results to {analysis_file_path}")

        if batch_type == "watermark_resize":
            current_watermark += 1

        # Calculate elapsed time and estimate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time * (total_images - (i + 1))
        print(
            f"Processed image pair {i + 1}/{total_images} in {elapsed_time:.2f} seconds. "
            f"Estimated time remaining: {remaining_time / 60:.2f} minutes."
        )

    return results


def analyse(
    original_image: Tensor,
    watermarked_image: Tensor,
    extracted_watermark: Tensor,
    styled_watermarked_image: Tensor,
    styled_extracted_watermark: Tensor,
    styled: Tensor,
):
    """
    Analyse the differences between the original and watermarked images.

    Args:
        watermarked_image: The watermarked image tensor.
        extracted_watermark: The extracted watermark tensor.
        styled_watermarked_image: The styled watermarked image tensor.
        styled_extracted_watermark: The styled extracted watermark tensor.
        styled: The styled image tensor.

    Returns:
        None
    """
    original_image = original_image.clone().cpu()
    watermarked_image = watermarked_image.clone().cpu()
    extracted_watermark = extracted_watermark.clone().cpu()
    styled_watermarked_image = styled_watermarked_image.clone().cpu()
    styled_extracted_watermark = styled_extracted_watermark.clone().cpu()
    styled = styled.clone().cpu()

    # Calculate pixel difference
    pixel_diff_styled = pixel_difference_float(
        watermarked_image, styled_watermarked_image
    ).item()
    pixel_diff_extracted = pixel_difference_float(
        extracted_watermark, styled_extracted_watermark
    ).item()
    pixel_diff_styled_vs_watermarked = pixel_difference_float(
        styled, styled_watermarked_image
    ).item()

    # Calculate PSNR
    psnr_styled = PSNR(watermarked_image, styled_watermarked_image)
    psnr_extracted = PSNR(extracted_watermark, styled_extracted_watermark)
    psnr_styled_vs_watermarked = PSNR(styled, styled_watermarked_image)

    # Calculate perceptual difference
    perceptual_diff_styled = perceptualDifference(
        watermarked_image, styled_watermarked_image
    ).item()
    perceptual_diff_extracted = perceptualDifference(
        extracted_watermark, styled_extracted_watermark
    ).item()
    perceptual_diff_styled_vs_watermarked = perceptualDifference(
        styled, styled_watermarked_image
    ).item()

    # Calculate structural difference
    structural_diff_styled = structuralDifference(
        watermarked_image, styled_watermarked_image
    )
    structural_diff_extracted = structuralDifference(
        extracted_watermark, styled_extracted_watermark
    )
    structural_diff_styled_vs_watermarked = structuralDifference(
        styled, styled_watermarked_image
    )

    # Print results
    print(f"Pixel Difference (Watermarked vs Styled Watermarked): {pixel_diff_styled}")
    print(f"Pixel Difference (Extracted vs Styled Extracted): {pixel_diff_extracted}")
    print(
        f"Pixel Difference (Styled vs Styled Watermarked): {pixel_diff_styled_vs_watermarked}"
    )

    print(f"PSNR (Watermarked vs Styled Watermarked): {psnr_styled}")
    print(f"PSNR (Extracted vs Styled Extracted): {psnr_extracted}")
    print(f"PSNR (Styled vs Styled Watermarked): {psnr_styled_vs_watermarked}")

    print(
        f"Perceptual Difference (Watermarked vs Styled Watermarked): {perceptual_diff_styled}"
    )
    print(
        f"Perceptual Difference (Extracted vs Styled Extracted): {perceptual_diff_extracted}"
    )
    print(
        f"Perceptual Difference (Styled vs Styled Watermarked): {perceptual_diff_styled_vs_watermarked}"
    )

    print(
        f"Structural Difference (Watermarked vs Styled Watermarked): {structural_diff_styled}"
    )
    print(
        f"Structural Difference (Extracted vs Styled Extracted): {structural_diff_extracted}"
    )
    print(
        f"Structural Difference (Styled vs Styled Watermarked): {structural_diff_styled_vs_watermarked}"
    )

    return (
        pixel_diff_styled,
        pixel_diff_extracted,
        pixel_diff_styled_vs_watermarked,
        psnr_styled,
        psnr_extracted,
        psnr_styled_vs_watermarked,
        perceptual_diff_styled,
        perceptual_diff_extracted,
        perceptual_diff_styled_vs_watermarked,
        structural_diff_styled,
        structural_diff_extracted,
        structural_diff_styled_vs_watermarked,
    )


def test():
    print("Hello from copyright-dissertation!")

    watermarkTensor: Tensor = preprocessImage(WATERMARK, DEVICE).to(DEVICE)
    contentTensor: Tensor = preprocessImage(CONTENT_IMAGES_LIST[0], DEVICE).to(DEVICE)
    styleTensor: Tensor = preprocessImage(STYLE_IMAGES_LIST[2], DEVICE).to(DEVICE)

    dwt_alphas = [0.0, 0.0, 0.0, 1.0]
    dct_alpha = 0

    # perform DCT DWT watermark embedding
    [finalTensor, extracted, _, _, _, _] = embedWatermark(
        contentTensor,
        watermarkTensor,
        DWT_alphas=[
            torch.tensor(dwt_alphas[0]).cuda(),
            torch.tensor(dwt_alphas[1]).cuda(),
            torch.tensor(dwt_alphas[2]).cuda(),
            torch.tensor(dwt_alphas[3]).cuda(),
        ],
        DCT_alpha=torch.tensor(dct_alpha).cuda(),
        display=True,
    )

    # Save the final tensor as an image
    final_image: Image = tensorToImage(finalTensor)
    final_image.save("final_tensor_image.png")

    percep = perceptualDifference(finalTensor, contentTensor, True)

    print("Final tensor image saved as 'final_tensor_image.png'")

    input("Press Enter to continue...")
    return


if __name__ == "__main__":

    test()
    # batch_watermark(
    #     batch_type="same content",
    #     content_image=Path(DATA_PATH / "lena.jpg"),
    #     ext_path=Path("lena"),
    #     epochs=10,
    #     count=50,
    # )

    # batch_watermark(batch_type="same style", epochs=25, count=10)
    # batch_watermark(batch_type="same content", epochs=25, count=10)
    # batch_watermark(batch_type="random", epochs=25, count=10)
