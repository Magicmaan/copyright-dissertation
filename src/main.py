from PIL import Image
from pathlib import Path
from GAN import *
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
from GAN import adversarialLoss
from GAN import Discriminator
from GAN import discriminatorLoss
from GAN import Generator
from GAN import PerceptualLoss
from util.NST import performNST
from util.comparison import *
from util.dctdwt import (
    embedWatermark,
    embedWatermarkDCT,
    embedWatermarkDWT,
    extractWatermarkDCT,
    extractWatermarkDWT,
)
import torch
from util.debug import displayImageTensors
from util.image import imageToTensor, preprocessImage, tensorToImage
import os

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
DISCRIMINATOR = Discriminator()

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


# WORK IN PROGRESS
def train(watermarkTensor: Tensor, contentTensor: Tensor) -> None:
    for epoch in range(EPOCHS):

        [watermarkedTensor, extractedWatermark] = GENERATOR.forward(
            contentTensor, watermarkTensor
        )

        styledTensor = performNST(
            contentTensor, styledTensor, outputPath="output/gen_latest.png"
        )
        # styledTensor = styleTransfer(watermarkedTensor, styleTensor)

        # TODO
        styledExtractedWatermark = torch.Tensor()
        # styledExtractedWatermark = extractWatermarkNST()

        # train the generator
        generatorOptimiser.zero_grad()
        # get loss between the generated image and the content image (visual)
        pLoss = perceptualDifference(watermarkedTensor, contentTensor)
        # get watermark loss
        wLoss = MSEDifference(watermarkedTensor, styledExtractedWatermark)
        # adversarial loss
        # learns to keep watermark after NST
        # high loss = watermark is not present in the image
        # low loss = watermark is present in the image
        aLoss = adversarialLoss(DISCRIMINATOR, watermarkedTensor)

        # discriminator loss
        # learns to differentiate between pre-NST and post-NST images
        # high loss = watermark is present in the image
        # low loss = watermark is present in the image
        discriminatorLoss = discriminatorLoss(
            DISCRIMINATOR, watermarkedTensor, styledTensor
        )

        # get total loss for the generator
        generatorLoss = (
            WEIGHTS["perceptual"] * pLoss
            + WEIGHTS["watermark"] * wLoss
            + WEIGHTS["adversarial"] * aLoss
        )
        generatorLoss.backward()
        generatorOptimiser.step()


def main():
    print("Hello from copyright-dissertation!")

    watermarkTensor: Tensor = preprocessImage(WATERMARK, DEVICE)
    contentTensor: Tensor = preprocessImage(CONTENT_IMAGES_LIST[0], DEVICE)
    styleTensor: Tensor = preprocessImage(STYLE_IMAGES_LIST[0], DEVICE)

    # input("Press Enter to continue...")
    # performNST()

    # TEST

    # # Load the generated image for watermark extraction
    # generatedImagePath = DATA_PATH / "test" / "gen_200.png"
    # assert (
    #     generatedImagePath.exists()
    # ), "Generated image not found at the specified path."

    # generatedImage: Image = Image.open(generatedImagePath)
    # generatedTensor: Tensor = preprocessImage(generatedImage, DEVICE)

    # # Perform DST watermark extraction
    # extractedWatermark = extractWatermarkDCT(
    #     contentTensor, generatedTensor, alpha=0.0001
    # )

    # displayImageTensors(extractedWatermark, titles=["Extracted Watermark (DWT)"])

    # # END TEST

    # Display the extracted watermark
    # displayImageTensors(extractedWatermark, titles=["Extracted Watermark"])

    print(contentTensor)
    print(styleTensor)

    # DWT DCT alpha values
    # controls the strength of the watermark
    # higher values = stronger / more visible watermark
    DWTAlpha = [0.6, 0.6, 0.05, 0.1]
    DCTAlpha = 0.1

    # perform DCT DWT watermark embedding
    [finalTensor, extracted, _, _, _, _] = embedWatermark(
        contentTensor,
        watermarkTensor,
        alphasDWT=DWTAlpha,
        alphaDCT=DCTAlpha,
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
    perceptualDiff = perceptualDifference(contentTensor, finalTensor, False)
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

    displayImageTensors(
        pixelDiff,
        extracted,
        titles=[
            "Pixel Difference",
            "Extracted Watermark",
        ],
    )

    styled = performNST(
        finalTensor.to(DEVICE, torch.float),
        styleTensor.to(DEVICE, torch.float),
        iterations=10,
    )
    displayImageTensors(styled, titles=["Styled Tensor"])

    NSTPerceptualDiff = perceptualDifference(
        finalTensor.cpu(),
        styled.cpu(),
        True,
    )

    NSTDWTExtractedWatermark = extractWatermarkDWT(
        contentTensor.cpu(),
        styled.cpu(),
        alphas=DWTAlpha,
    )
    NSTDCTExtractedWatermark = extractWatermarkDCT(
        contentTensor.cpu(),
        styled.cpu(),
        alpha=0.0001,
    )

    pixelDiffStyledFinal = pixelDifference(styled.cpu(), finalTensor.cpu())

    displayImageTensors(
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
    main()
