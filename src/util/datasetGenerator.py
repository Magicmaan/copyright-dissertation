import torchvision

import os
import random
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from dctdwt import embedWatermark

from image import imageToTensor

CURRENT_FILE_PATH = Path(__file__).resolve()
TRAINING_PATH = CURRENT_FILE_PATH.parent.parent.parent / "data" / "training"
print(TRAINING_PATH)
assert TRAINING_PATH.exists(), "The training path does not exist."

IMAGES_PATH = TRAINING_PATH / "content"
WATERMARK_PATH = TRAINING_PATH / "watermark" / "watermark.jpg"

OUTPUT_PATH = TRAINING_PATH / "output"


WATERMARK = imageToTensor(Image.open(WATERMARK_PATH))


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
        return Image.open(imagePath).convert("RGB")


def generateDataset(count: int = 50) -> int:
    dataset = ImageDataset(IMAGES_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, image in enumerate(dataloader):
        if i >= count:
            break

        tensor = imageToTensor(image[0])

        [watermarked, extracted, _, _, _, _, _] = embedWatermark(
            tensor,
            WATERMARK,
            alphaDWT=[0.1, 0.1, 0.1, 0.1],
            DCT_alpha=[0.1, 0.1, 0.1, 0.1],
            display=False,
        )

        watermarkImage = Image.fromarray(
            (watermarked.squeeze(0).numpy() * 255).astype("uint8")
        )
        outputPath = OUTPUT_PATH / f"watermarked_{i}.jpg"

        watermarkImage.save(outputPath)
        print(f"Saved watermarked image to {outputPath}")

        # Process each image here (e.g., apply transformations, save output, etc.)


if __name__ == "__main__":
    generateDataset(50)
