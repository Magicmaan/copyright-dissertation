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
import numpy as np
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor


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


def extractFeatures(model, generatedImage, layers):
    # https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573
    # heavily inspired by the above link

    # Extract convolution layers from the model
    children = list(model.children())
    convolutionLayers: list[torch.nn.Conv2d] = []
    for layer in layers:
        convolutionLayers.append(children[0][layer])

    featureMaps: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    # Create image from tensor and add to images list
    tempImage = generatedImage
    for layer in convolutionLayers:
        tempImage = layer(tempImage)
        featureMaps.append(tempImage)

    # turn tensor into grayscale image
    for fMap in featureMaps:
        fMap = fMap.squeeze(0)
        gray_scale = torch.sum(fMap, 0)
        gray_scale = gray_scale / fMap.shape[0]
        images.append(gray_scale.data.cpu().numpy())

    return images, featureMaps


def numpyListToTensorList(images: list[np.ndarray]) -> list[torch.Tensor]:
    """
    Convert a list of numpy arrays to a list of PyTorch tensors.

    :param images: List of numpy arrays to convert.
    :return: List of PyTorch tensors.
    """
    return [torch.from_numpy(image).float() for image in images]
