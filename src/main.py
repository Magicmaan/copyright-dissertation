from PIL import Image
from pathlib import Path
from GAN import *
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
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
from util.texture import imageToTensor, preprocessImage, tensorToImage

torch.manual_seed(0)  # for reproducibility

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

# generator = Generator()
# discriminator = Discriminator()

generator_optimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
discriminator_optimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))


# hyper parameters for training
EPOCHS = 100

WEIGHTS = {"watermark": 1, "adversarial": 1, "perceptual": 1}

WATERMARK_LOSS = WatermarkLoss()
ADVERSARIAL_LOSS = AdversarialLoss()
DISCRIMINATOR_LOSS = DiscriminatorLoss()
TOTAL_LOSS = TotalLoss(
    WEIGHTS["perceptual"], WEIGHTS["watermark"], WEIGHTS["adversarial"]
)


def main():
    print("Hello from copyright-dissertation!")

    watermarkTensor: Tensor = preprocessImage(WATERMARK)
    contentTensor: Tensor = preprocessImage(CONTENT_IMAGES_LIST[0])
    styleTensor: Tensor = preprocessImage(STYLE_IMAGES_LIST[0])

    # TEST

    # Load the generated image for watermark extraction
    generatedImagePath = DATA_PATH / "test" / "gen_200.png"
    assert (
        generatedImagePath.exists()
    ), "Generated image not found at the specified path."

    generatedImage: Image = Image.open(generatedImagePath)
    generatedTensor: Tensor = preprocessImage(generatedImage)

    # Perform DST watermark extraction
    extractedWatermark = extractWatermarkDCT(
        contentTensor, generatedTensor, alpha=0.0001
    )

    displayImageTensors(extractedWatermark, titles=["Extracted Watermark (DWT)"])

    # END TEST

    # Display the extracted watermark
    displayImageTensors(extractedWatermark, titles=["Extracted Watermark"])

    print(contentTensor)
    print(styleTensor)

    # DWT DCT alpha values
    # controls the strength of the watermark
    # higher values = stronger / more visible watermark
    DWTAlpha = 0.001
    DCTAlpha = 0.0001

    [finalTensor, extracted, _, _, _, _] = embedWatermark(
        contentTensor,
        watermarkTensor,
        alphasDWT=[DWTAlpha, DWTAlpha, DWTAlpha * 40, DWTAlpha * 40],
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
    perceptualDiff = perceptualDifference(contentTensor, finalTensor)
    structuralDiff = structuralDifference(contentTensor, finalTensor)
    peakNoise = PSNR(contentTensor, finalTensor)
    print("after embedding watermark")

    # amplify the differences
    pixelDiff *= 255.0
    pixelDiff /= DWTAlpha
    MSEDiff *= 255.0
    print("Pixel Difference: ", pixelDiff)
    print("MSE Difference: ", MSEDiff)
    print("Perceptual Difference: ", perceptualDiff)
    print("Structural Difference: ", structuralDiff)
    print("Peak Noise: ", peakNoise)

    displayImageTensors(
        pixelDiff,
        titles=[
            "Pixel Difference",
            "MSE Difference",
        ],
    )
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
