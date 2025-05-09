import math
from torch import Tensor
import numpy as np

import matplotlib.pyplot as plt

from util.image import tensorToImage


def getVariableName(var: any) -> str:
    """
    Get the variable name as a string.

    :param: var: Variable to get the name of.

    :return: Name of the variable.
    """
    for name, value in globals().items():
        if value is var:
            return name
    return "Unknown"


def display_image_tensors(
    *tensors: Tensor | list[Tensor], titles: list[str] = None
) -> None:
    """
    Display multiple tensors or a list of tensors as images.

    :param: tensors: Tensors or a list of tensors to display as images.
    :param: titles: Optional list of titles for the images.
    """
    # Flatten the input in case a list of tensors is passed
    flattenedTensors = []
    for tensor in tensors:
        if isinstance(tensor, list):
            flattenedTensors.extend(tensor)
        else:
            flattenedTensors.append(tensor)

    numTensors = len(flattenedTensors)
    fig, axes = plt.subplots(1, numTensors, figsize=(numTensors * 5, 5))
    plt.ion()  # Turn on interactive mode for live updates
    if numTensors == 1:
        axes = [axes]

    for ax, tensor in zip(axes, flattenedTensors):
        image = tensorToImage(tensor)

        # Determine colormap based on image mode
        if image.mode == "L":
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        if titles is not None and titles:
            ax.set_title(titles.pop(0))
        else:
            ax.set_title(getVariableName(tensor))
        ax.axis("off")  # Hide axes for better visualization

    plt.show(block=False)  # Show the plot without blocking
    plt.pause(0.001)  # Pause to allow for live updates


def display_image_np(images: list[np.ndarray], titles: list[str] = None) -> None:
    numFeatures = len(images)
    cols = 4
    rows = math.ceil(numFeatures / cols)
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i in range(numFeatures):
        a = fig.add_subplot(rows, cols, i + 1)
        imgplot = plt.imshow(images[i])
        a.axis("off")
        # if titles is not None and titles:
        #     a.set_title(titles.pop(0))
        # else:
        #     a.set_title(f"Image Layer {i}")

    plt.show(block=False)  # Show the plot without blocking
    plt.pause(0.001)  # Pause to allow for live updates
