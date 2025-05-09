import math
import torch
from torch import Tensor

# importing the required libraries
from pathlib import Path
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import argparse


# vgg class to store model and features
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # convolution layers to be used
        self.req_features = ["0", "5", "10", "19", "28"]
        self.model = models.vgg19(models.VGG19_Weights.DEFAULT).features[:29]
        train_nodes, test_nodes = get_graph_node_names(self.model)

        print(f"Train nodes: {train_nodes}")
        print(f"Test nodes: {test_nodes}")

    def forward(self, image):
        features = []
        # extract layers from model
        for i, layer in enumerate(self.model):
            # activation of the layer will stored in x
            image = layer(image)
            # appending the activation of the selected layers and return the feature array
            if str(i) in self.req_features:
                features.append(image)

        return features

    def extract_features(
        self, tensor: torch.Tensor, extract_layers: list[int] = list(range(29))
    ) -> tuple[list[np.ndarray], list[torch.Tensor]]:
        """
        Extract features and corresponding grayscale images from the model.

        :param tensor: Input tensor to extract features from.
        :param layers: List of layer indices to extract features from.
        :return: Tuple containing a list of grayscale images and a list of feature maps.
        """
        # Extract convolution layers from the model
        children = list(self.model.children())
        convolutionLayers: list[torch.nn.Conv2d] = []
        for layer in extract_layers:
            convolutionLayers.append(children[layer])

        featureMaps: list[torch.Tensor] = []
        images: list[np.ndarray] = []

        # Create image from tensor and add to images list
        tempImage = tensor
        for layer in convolutionLayers:
            tempImage = layer(tempImage)
            featureMaps.append(tempImage)

        # Turn tensor into grayscale image
        for fMap in featureMaps:
            fMap = fMap.squeeze(0)
            grayScale = torch.sum(fMap, 0)
            grayScale = grayScale / fMap.shape[0]
            images.append(grayScale.data.detach().cpu().numpy())

        return images, featureMaps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vgg = VGG().to(device).eval()


def display_image_np(
    images: list[np.ndarray], titles: list[str] = None, save_path: str = None
) -> None:
    """
    Display or save a list of images.

    :param images: List of images as numpy arrays.
    :param titles: List of titles for the images.
    :param save_path: Path to save the images as a single figure. If None, the images are displayed.
    """
    num_features = len(images)
    cols = 4
    rows = math.ceil(num_features / cols)
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i in range(num_features):
        a = fig.add_subplot(rows, cols, i + 1)
        imgplot = plt.imshow(images[i])
        a.axis("off")
        if titles is not None and titles:
            a.set_title(titles.pop(0))
        else:
            a.set_title(f"Image Layer {i}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Images saved to {save_path}")
    else:
        plt.show(block=False)  # Show the plot without blocking
        plt.pause(0.001)  # Pause to allow for live updates


def perceptual_difference(
    image1: torch.Tensor,
    image2: torch.Tensor,
    display_features: bool = False,
    save_path: str = None,
) -> float:
    """
    Calculate the perceptual difference between two images feature maps using MSE.
    Args:
        image1: The first image.
        image2: The second image.
        display_features: Whether to display the feature maps of the images.
        save_path: Path to save the feature maps as images.
    Returns:
        float: The perceptual difference between the two images.
        (higher is worse, 0 is perfect match)
    """
    image1 = image1.clone().to(device)
    image2 = image2.clone().to(device)
    conv_layers = list(range(29))  # Include all layers

    image1_features_np, image1_features_tensor = vgg.extract_features(image1)
    image2_features_np, image2_features_tensor = vgg.extract_features(image2)

    if display_features:
        display_image_np(
            image1_features_np, conv_layers, save_path=save_path + "features_1.png"
        )

        display_image_np(
            image2_features_np, conv_layers, save_path=save_path + "features_2.png"
        )

    difference: list[torch.Tensor] = []
    for i in range(len(image1_features_tensor)):
        features1 = image1_features_tensor[i] * 10
        features2 = image2_features_tensor[i] * 10

        diff = (features1 - features2) * 100
        difference.append(diff)

    # get total MSE
    image1_features = torch.cat([f.flatten() for f in image1_features_tensor], dim=0)
    image2_features = torch.cat([f.flatten() for f in image2_features_tensor], dim=0)

    mse = torch.nn.functional.mse_loss(image1_features, image2_features).item()
    return mse


def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """
    Load an image from a file and convert it to a tensor.

    :param image_path: Path to the image file.
    :return: Image as a tensor.
    """
    assert Path(image_path).exists(), f"Image file {image_path} does not exist."
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run perceptual difference on two images."
    )
    parser.add_argument("image1_path", type=str, help="Path to the first image.")
    parser.add_argument("image2_path", type=str, help="Path to the second image.")
    parser.add_argument("--display", action="store_true", help="Display feature maps.")
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save feature maps."
    )

    args = parser.parse_args()

    image1_tensor = load_image_as_tensor(args.image1_path).to(device)
    image2_tensor = load_image_as_tensor(args.image2_path).to(device)

    result = perceptual_difference(
        image1_tensor, image2_tensor, args.display, args.save_path
    )
    print("Perceptual Difference:", result)
