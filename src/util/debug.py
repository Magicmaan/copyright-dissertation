from torch import Tensor
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

from util.texture import convert_tensor_to_image


def get_variable_name(var: any) -> str:
    """
    Get the variable name as a string.

    @param: var: Variable to get the name of.

    :return: Name of the variable.
    """
    for name, value in globals().items():
        if value is var:
            return name
    return "Unknown"


def display_image_tensors(*tensors: Tensor, titles: list[str] = None) -> None:
    """
    Display multiple tensors as images.

    @param: tensors: Tensors to display as images.
    """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(num_tensors * 5, 5))
    plt.ion()  # Turn on interactive mode for live updates
    if num_tensors == 1:
        axes = [axes]

    for ax, tensor in zip(axes, tensors):
        image = convert_tensor_to_image(tensor)

        # Determine colormap based on image mode
        if image.mode == "L":
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image)

        if titles is not None and titles:
            ax.set_title(titles.pop(0))
        else:
            ax.set_title(get_variable_name(tensor))
        ax.axis("off")  # Hide axes for better visualization

    plt.show(block=False)  # Show the plot without blocking
    plt.pause(0.001)  # Pause to allow for live updates
