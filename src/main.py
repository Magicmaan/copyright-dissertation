from PIL import Image
from pathlib import Path
from lossFunctions import *
from torch.optim import Adam

# load assets
data_path = Path("data")

# load watermark
watermark: Image = Image.open(data_path / "watermark.png")
assert watermark is not None, "Watermark not found."

# load content and style images
contentImages: list[Path] = list(data_path.glob("content/*.jpg"))
styleImages: list[Path] = list(data_path.glob("style/*.jpg"))
assert len(contentImages) > 0, "No content images found."
assert len(styleImages) > 0, "No style images found."


generator_optimiser = Adam()
discriminator_optimiser = Adam()


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


if __name__ == "__main__":
    main()
