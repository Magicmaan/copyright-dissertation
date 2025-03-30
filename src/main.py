from PIL import Image
from pathlib import Path
from lossFunctions import *
from torch.optim import Adam
from torch import Tensor
import torch.nn as nn
from util.dctdwt import (
    embed_watermark_DCT,
    embed_watermark_DWT,
    extract_watermark_DCT,
    extract_watermark_DWT,
)
from util.debug import display_image_tensors
from util.texture import convert_image_to_tensor, preprocess_image

# load assets
data_path = Path("data")

# load watermark
watermark: Image = Image.open(data_path / "watermark.jpg")
assert watermark is not None, "Watermark not found."


# load content and style images
contentImages: list[Path] = list(data_path.glob("content/*.jpg"))
styleImages: list[Path] = list(data_path.glob("style/*.jpg"))
assert len(contentImages) > 0, "No content images found."
assert len(styleImages) > 0, "No style images found."

contentImages = [Image.open(image) for image in contentImages]
styleImages = [Image.open(image) for image in styleImages]

model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

# generator = Generator()
# discriminator = Discriminator()

generator_optimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))
discriminator_optimiser = Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999))


# hyper parameters for training
epochs = 100

weights = {"watermark": 1, "adversarial": 1, "perceptual": 1}

watermark_loss = WatermarkLoss()
adversarial_loss = AdversarialLoss()
discriminator_loss = DiscriminatorLoss()
total_loss = TotalLoss(
    weights["perceptual"], weights["watermark"], weights["adversarial"]
)


def main():
    print("Hello from copyright-dissertation!")

    watermarkTensor: Tensor = preprocess_image(watermark)
    contentTensor: Tensor = preprocess_image(contentImages[0])
    styleTensor: Tensor = preprocess_image(styleImages[0])

    print(contentTensor)
    print(styleTensor)

    # DWT DCT alpha values
    # controls the strength of the watermark
    # higher values = stronger / more visible watermark
    DWTAlpha = 1
    DCTAlpha = 1

    watermarkedTensorDWT = embed_watermark_DWT(contentTensor, watermarkTensor, DWTAlpha)
    extracted_watermarkDWT = extract_watermark_DWT(
        contentTensor, watermarkedTensorDWT, DWTAlpha
    )

    finalDCT = embed_watermark_DCT(
        watermarkedTensorDWT, extracted_watermarkDWT, DCTAlpha
    )
    final_extracted_watermarkDCT = extract_watermark_DCT(
        contentTensor, finalDCT, DCTAlpha
    )

    display_image_tensors(
        contentTensor,
        watermarkTensor,
        watermarkedTensorDWT,
        extracted_watermarkDWT,
        finalDCT,
        final_extracted_watermarkDCT,
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
