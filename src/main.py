from PIL import Image
import numpy as np
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

# load watermark
WATERMARK: Image = Image.open(DATA_PATH / "watermark.jpg")
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


model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

GENERATOR = Generator()
DISCRIMINATOR = DiscriminatorExt()

generatorOptimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.5, 0.999))
discriminatorOptimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.5, 0.999))


# hyper parameters for training
EPOCHS = 100

WEIGHTS = {"watermark": 0.3, "adversarial": 0.3, "perceptual": 0.3}

WATERMARK_LOSS = WatermarkLoss()


PERCEPTUAL_LOSS = PerceptualLoss()
TOTAL_LOSS = TotalLoss(
    WEIGHTS["perceptual"], WEIGHTS["watermark"], WEIGHTS["adversarial"]
)


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
    def __init__(self, imagesPath: Path):
        self.imagesPath = imagesPath
        self.imageFiles = [f for f in imagesPath.glob("*.jpg") if f.is_file()]
        assert self.imageFiles, "No .jpg files found in the specified path."

    def __len__(self) -> int:
        return len(self.imageFiles)

    def __getitem__(self, index: int) -> Image.Image:
        imagePath = self.imageFiles[index]
        assert imagePath.exists(), f"File {imagePath} does not exist."
        return imageToTensor(Image.open(imagePath).convert("RGB"))


def makeDataset(count: int = 50):
    TRAINING_PATH = Path("data") / "training"
    IMAGES_PATH = TRAINING_PATH / "content"
    WATERMARK_PATH = TRAINING_PATH / "watermark" / "watermark.jpg"
    OUTPUT_PATH = TRAINING_PATH / "output"
    assert IMAGES_PATH.exists(), "The images path does not exist."
    assert WATERMARK_PATH.exists(), "The watermark path does not exist."

    WATERMARK = imageToTensor(Image.open(WATERMARK_PATH))
    assert WATERMARK is not None, "Watermark not found."
    dataset = ImageDataset(IMAGES_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, image in enumerate(dataloader):
        if i >= count:
            break

        tensor = image[0]

        [watermarked, extracted, _, _, _, _] = embedWatermark(
            tensor,
            WATERMARK,
            DWT_alphas=[0.001, 0.0001, 0.0001, 0.0001],
            DCT_alpha=0.0001,
            display=False,
        )

        watermarkImage = tensorToImage(watermarked)
        outputPath = OUTPUT_PATH / f"watermarked_{i}.jpg"

        watermarkImage.save(outputPath)
        print(f"Saved watermarked image to {outputPath}")

        # Process each image here (e.g., apply transformations, save output, etc.)


# Training loop
def train_adversarial_watermarking(
    content_images, watermark, style_images, epochs=10, device="cuda"
):
    # Learnable alphas (initialized small)
    alphas = torch.nn.Parameter(
        torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05], requires_grad=True, device=device)
    )

    discriminator = Discriminator().to(device)
    optimizer_G = Adam([alphas], lr=1e-3)
    optimizer_D = Adam(discriminator.parameters(), lr=1e-4)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    for epoch in range(epochs):
        for i, (content_img, style_img) in enumerate(zip(content_images, style_images)):
            content_img = content_img.to(device)
            style_img = style_img.to(device)

            # --- Generator step ---
            optimizer_G.zero_grad()

            [wm_img, extracted, _, _, _, _] = embedWatermark(
                content_img, watermark, *alphas
            )
            stylized_img = performNST(wm_img, style_img)

            recovered_wm = extract_watermark(stylized_img)

            # Losses
            watermark_loss = mse_loss(recovered_wm, watermark)
            perceptual_loss = mse_loss(content_img, wm_img)

            pred = discriminator(stylized_img)
            adversarial_loss = -torch.mean(torch.log(pred + 1e-8))

            total_loss = (
                1.0 * perceptual_loss + 2.0 * watermark_loss + 1.0 * adversarial_loss
            )

            total_loss.backward()
            optimizer_G.step()

            # --- Discriminator step ---
            optimizer_D.zero_grad()
            pred_real = discriminator(wm_img.detach())
            pred_fake = discriminator(stylized_img.detach())

            loss_D = -torch.mean(torch.log(pred_fake + 1e-8)) + torch.mean(
                torch.log(1 - pred_real + 1e-8)
            )
            loss_D.backward()
            optimizer_D.step()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch}, Step {i}: Total Loss {total_loss.item():.4f}, Watermark {watermark_loss.item():.4f}"
                )

    return alphas.detach().cpu()


def plot_results(
    epochs: int,
    total_loss: list[float],
    pixel_loss: list[float],
    watermark_loss: list[float],
    adversarial_loss: list[float],
    dct_alpha: list[float],
    dwt_alpha: list[list[float, float, float, float]],
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
        plt.savefig("training_progress_log_scale.png")
        plt.draw()
        plt.pause(0.001)
    except Exception as e:
        print(f"Error plotting results: {e}")


def train(
    content_tensor: Tensor, watermark_tensor: Tensor, style_tensor: Tensor, epochs=100
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

    # initialise adversary
    adversary = WatermarkDetector().to(DEVICE)
    adversary_optimiser = Adam(adversary.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Initialize alpha parameters with requires_grad=True - using smaller initial values
    # dwt_alphas = nn.Parameter(
    #     torch.tensor([0.1, 0.1, 0.1, 0.1], device=DEVICE, requires_grad=True)
    # )
    # dwt_alphas = Tensor([0.1, 0.1, 0.1, 0.1]).to(DEVICE)
    dwt_a1 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a2 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a3 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dwt_a4 = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))
    dct_alpha = nn.Parameter(torch.tensor(0.1, device=DEVICE, requires_grad=True))

    # Create optimizer with appropriate learning rate
    optimiser = Adam([dwt_a1, dwt_a2, dwt_a3, dwt_a4, dct_alpha], lr=0.01)

    # Loss weights - giving more weight to the watermark loss
    parameters = {
        "pixel": 0.3,  # Visual quality preservation
        "adversarial": 1.5,  # Adversarial loss for watermark detection
        "dwt_weights": 0.8,  # DWT alphas weight
    }

    # Arrays to track loss and alpha values
    losses = []
    dwt_alpha_history: list[list[float, float, float, float]] = []
    dct_alpha_history: list[float] = []

    history = {
        "total_loss": [],
        "pixel_loss": [],
        "watermark_loss": [],
        "adversarial_loss": [],
        "dct_alpha": [],
        "dwt_alpha": [],
    }

    # dev stuff to cancel if values dont change
    stall_count = 0
    stall_count_max = 2

    print("Beginning training...")
    for epoch in range(epochs):
        # Reset gradients
        optimiser.zero_grad()
        adversary_optimiser.zero_grad()
        # dct_alpha.grad
        dct_alpha.requires_grad = True
        dwt_a1.requires_grad = True
        dwt_a2.requires_grad = True
        dwt_a3.requires_grad = True
        dwt_a4.requires_grad = True

        # print(f"gradient: {dct_alpha.grad}")
        # print(f"gradient: {dwt_alphas.grad}")

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

        # get adversary prediction for if watermark exists
        prediction = adversary(content_tensor, styled)

        # the ground truth ( watermark 100% exists) is 1.0, so initialise target for adversary to aim for
        label = torch.ones_like(prediction, device=DEVICE)

        adversary_loss = torch.nn.functional.binary_cross_entropy(prediction, label)
        print("Watermark Adversary Loss: ", adversary_loss.item())
        # Calculate losses
        pixel_loss = torch.nn.functional.mse_loss(content_tensor, watermarked)
        # watermark_loss = torch.nn.functional.mse_loss(watermark_tensor, extracted)
        # pixel_loss.requires_grad = True
        # watermark_loss.requires_grad = True
        # We want to minimize pixel loss and maximize watermark quality
        # (minimizing negative watermark loss = maximizing extraction quality)
        # fmt: off
        total_loss = (
            parameters["pixel"] * pixel_loss
            # - parameters["watermark"] * (-watermark_loss)
            * (parameters["adversarial"] * adversary_loss)
        )
        # fmt: on

        # total_loss.requires_grad = True
        # Backpropagate
        total_loss.backward()

        dct_alpha.backward()

        def relative_decay_loss(alphas: list[torch.Tensor]) -> torch.Tensor:
            """
            Adds a penalty if DWT alphas are not strictly decreasing.
            Encourages dwtAlphas[0] > dwtAlphas[1] > ... > dwtAlphas[n]
            """
            penalty = 0.0
            for i in range(1, len(alphas)):
                target_alpha = alphas[i - 1] * parameters["dwt_weights"]
                penalty += (alphas[i] - target_alpha) * 2

            print("Penalty: ", penalty.item())
            return penalty

        # add weighting to dwt alphas to ensure model understands the weighting for each alpha
        # this is done since the lower dwt alphas embed more into style and less into content
        # a1 > a2 > a3 > a4
        regLoss = relative_decay_loss([dwt_a1, dwt_a2, dwt_a3, dwt_a4])
        # print(f"regLoss: {regLoss.item()}")
        total_loss += regLoss

        dwt_a1.backward()
        dwt_a2.backward()
        dwt_a3.backward()
        dwt_a4.backward()
        # dwt_alphas.backward()

        # Verify gradients exist before optimization step
        if dwt_a1.grad is None or dct_alpha.grad is None:
            print(f"Epoch {epoch+1}: No gradients - gradient flow issue detected")
            print(f"dwt_alphas.grad: {dwt_a1.grad}")
            print(f"dct_alpha.grad: {dct_alpha.grad}")

        # Update parameters
        optimiser.step()
        adversary_optimiser.step()

        # Clamp alphas to positive values (important to keep watermarking stable)
        with torch.no_grad():
            dwt_a1.data.clamp_(min=0.00001, max=0.1)
            dwt_a2.data.clamp_(min=0.00001, max=0.1)
            dwt_a3.data.clamp_(min=0.00001, max=0.1)
            dwt_a4.data.clamp_(min=0.00001, max=0.1)
            dct_alpha.data.clamp_(min=0.00001, max=0.1)

        # dev functionality to cancel if losses are not changing
        if (history["total_loss"] and len(history["total_loss"]) > 0) and (
            len(history["total_loss"]) > 1
        ):

            if total_loss.item() == history["total_loss"][-1]:
                stall_count += 1
                print(f"Loss stalled {stall_count} times")
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
        history["watermark_loss"].append(adversary_loss.item())
        history["adversarial_loss"].append(adversary_loss.item())
        history["dct_alpha"].append(dct_alpha.item())
        history["dwt_alpha"].append(
            [dwt_a1.item(), dwt_a2.item(), dwt_a3.item(), dwt_a4.item()]
        )
        # losses.append(total_loss.item())
        # dwt_alpha_history.append(
        #     [dwt_a1.item(), dwt_a2.item(), dwt_a3.item(), dwt_a4.item()]
        # )
        # dct_alpha_history.append(dct_alpha.item())

        # Print progress regularly
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], "
                f"Total Loss: {total_loss.item():.6f}, "
                f"Pixel Loss: {pixel_loss.item():.6f}, "
                # f"Watermark Loss: {watermark_loss.item():.6f}, "
                f"DWT Alphas: { dwt_a1.item()}, {dwt_a2.item()}, {dwt_a3.item()}, {dwt_a4.item()}], "
                f"DCT Alpha: {dct_alpha.item():.6f}"
            )

    # Plot the training progress
    if epochs > 0:
        print("epoch history")
        print(dct_alpha_history)
        print(dwt_alpha_history)
        # try:
        #     import matplotlib.pyplot as plt

        plot_results(
            epochs=epochs,
            total_loss=history["total_loss"],
            pixel_loss=history["pixel_loss"],
            watermark_loss=history["watermark_loss"],
            adversarial_loss=history["adversarial_loss"],
            dct_alpha=history["dct_alpha"],
            dwt_alpha=history["dwt_alpha"],
        )
        #     plt.ion()  # Turn on interactive mode
        #     plt.figure(figsize=(12, 8))

        #     # Plot loss
        #     plt.subplot(2, 1, 1)
        #     plt.plot(range(epochs), losses, label="Total Loss", color="red")
        #     plt.yscale("log")  # Use log scale for loss
        #     plt.title("Loss over epochs (Log Scale)")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Loss (Log Scale)")
        #     plt.legend()

        #     # Plot alphas
        #     plt.subplot(2, 1, 2)
        #     plt.plot(range(epochs), dct_alpha_history, label="DCT Alpha", color="blue")

        #     # Define colors for DWT alphas
        #     dwt_colours = ["orange", "green", "purple", "brown"]

        #     # Plot each DWT alpha separately with its own color
        #     for i in range(4):
        #         plt.plot(
        #             range(epochs),
        #             [alpha[i] for alpha in dwt_alpha_history],
        #             alpha=1 / (i + 1),
        #             color=dwt_colours[i],
        #             linestyle="dotted",
        #             linewidth=2,
        #             label=f"DWT Alpha {i+1}",
        #         )

        #     plt.yscale("log")  # Use log scale for alphas
        #     plt.title("Alpha Values over epochs (Log Scale)")
        #     plt.xlabel("Epoch")
        #     plt.ylabel("Alpha Value (Log Scale)")
        #     plt.legend()

        #     plt.tight_layout()
        #     plt.savefig("training_progress_log_scale.png")
        #     plt.draw()
        #     plt.pause(0.001)
        # except Exception as e:
        #     print(f"Error plotting results: {e}")

    print("Training complete.")
    # print(f"Final DWT Alphas: {dwt_alphas.detach().cpu().numpy()}")
    print(f"Final DCT Alpha: {dct_alpha.item()}")
    print(
        f"Final DWT Alphas: {dwt_a1.item()}, {dwt_a2.item()}, {dwt_a3.item()}, {dwt_a4.item()}"
    )

    return [
        dwt_a1.detach().item(),
        dwt_a2.detach().item(),
        dwt_a3.detach().item(),
        dwt_a4.detach().item(),
    ], dct_alpha.item()


def main():
    print("Hello from copyright-dissertation!")

    watermarkTensor: Tensor = preprocessImage(WATERMARK, DEVICE).to(DEVICE)
    contentTensor: Tensor = preprocessImage(CONTENT_IMAGES_LIST[0], DEVICE).to(DEVICE)
    styleTensor: Tensor = preprocessImage(STYLE_IMAGES_LIST[2], DEVICE).to(DEVICE)

    dwt_alphas, dct_alpha = train(
        contentTensor, watermarkTensor, styleTensor, epochs=200
    )

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

    print("Final tensor image saved as 'final_tensor_image.png'")

    input("Press Enter to continue...")
    return
    print(contentTensor)
    print(styleTensor)

    # DWT DCT alpha values
    # controls the strength of the watermark
    # higher values = stronger / more visible watermark
    DWTAlpha = [0.01, 0.01, 0.01, 0.01]
    DCTAlpha = 0.01

    # perform DCT DWT watermark embedding
    [finalTensor, extracted, _, _, _, _] = embedWatermark(
        contentTensor,
        watermarkTensor,
        DWT_alphas=DWTAlpha,
        DCT_alpha=DCTAlpha,
        display=True,
    )

    # Save the final image to a file
    finalImage: Image = tensorToImage(finalTensor)
    outputPath = DATA_PATH / "output" / "final_image.jpg"
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    finalImage.save(outputPath)
    print(f"Final image saved to {outputPath}")

    pixelDiff = pixelDifference(contentTensor, finalTensor)
    MSEDiff = MSEDifference(contentTensor, imageToTensor(finalImage))
    perceptualDiff = perceptualDifference(contentTensor, finalTensor, True)
    structuralDiff = structuralDifference(contentTensor, finalTensor)
    peakNoise = PSNR(contentTensor, finalTensor)
    print("after embedding watermark")

    # amplify the differences
    pixelDiff *= 255.0
    pixelDiff /= DWTAlpha[0]
    MSEDiff *= 255.0
    print("Pixel Difference: ", pixelDiff)
    print("MSE Difference: ", MSEDiff)
    print("Perceptual Difference: ", perceptualDiff)
    print("Structural Difference: ", structuralDiff)
    print("Peak Noise: ", peakNoise)

    display_image_tensors(
        pixelDiff,
        extracted,
        titles=[
            "Pixel Difference",
            "Extracted Watermark",
        ],
    )

    input("Press Enter to continue...")
    # Perform Neural Style Transfer (NST)
    styled = performNST(
        finalTensor.to(DEVICE, torch.float),
        styleTensor.to(DEVICE, torch.float),
        iterations=200,
    )
    display_image_tensors(styled, titles=["Styled Tensor"])

    NSTPerceptualDiff = perceptualDifference(
        finalTensor.cpu(),
        styled.cpu(),
        True,
    )

    NSTDWTExtractedWatermark = extract_watermark_dwt(
        contentTensor.cpu(),
        styled.cpu(),
        alphas=DWTAlpha,
    )
    NSTDCTExtractedWatermark = extract_watermark_dct(
        contentTensor.cpu(),
        styled.cpu(),
        alpha=0.0001,
    )

    pixelDiffStyledFinal = pixelDifference(styled.cpu(), finalTensor.cpu())

    display_image_tensors(
        NSTDCTExtractedWatermark,
        NSTDWTExtractedWatermark,
        pixelDiffStyledFinal,
        titles=[
            "DCT Extracted Watermark after NST",
            "DWT Extracted Watermark after NST",
            "Pixel Difference: Styled vs Final",
        ],
    )
    print("NST Perceptual Difference: ", NSTPerceptualDiff)

    input("Press Enter to Close...")


if __name__ == "__main__":
    # makeDataset(300)
    main()
