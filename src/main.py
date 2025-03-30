from PIL import Image
from pathlib import Path
from lossFunctions import *
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
from util.dctdwt import (
    embedWatermarkDCT,
    embedWatermarkDWT,
    extractWatermarkDCT,
    extractWatermarkDWT,
)
from util.debug import displayImageTensors
from util.texture import convert_image_to_tensor, preprocessImage

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

    print(contentTensor)
    print(styleTensor)

    # DWT DCT alpha values
    # controls the strength of the watermark
    # higher values = stronger / more visible watermark
    DWTAlpha = 1
    DCTAlpha = 1

    watermarkedDWT = embedWatermarkDWT(contentTensor, watermarkTensor, DWTAlpha)
    extractedWatermarkDWT = extractWatermarkDWT(contentTensor, watermarkedDWT, DWTAlpha)

    watermarkedDWT = embedWatermarkDCT(watermarkedDWT, extractedWatermarkDWT, DCTAlpha)
    extractedWatermarkDCT = extractWatermarkDCT(contentTensor, watermarkedDWT, DCTAlpha)

    displayImageTensors(
        contentTensor,
        watermarkTensor,
        watermarkedDWT,
        extractedWatermarkDWT,
        watermarkedDWT,
        extractedWatermarkDCT,
        titles=[
            "Content Image",
            "Watermark",
            "Watermarked Image DWT",
            "Extracted Watermark DWT",
            "Final Watermarked Image DCT",
            "Final Extracted Watermark DCT",
        ],
    )

    print("after embedding watermark")

    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
