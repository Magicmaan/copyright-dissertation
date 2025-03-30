from torch import Tensor
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

from util.texture import convert_tensor_to_image


def display_image_tensors(*tensors: Tensor, titles: list[str] = None) -> None:
    """
    Display multiple tensors as images.

    @param: tensors: Tensors to display as images.
    """
    num_tensors = len(tensors)
    fig, axes = plt.subplots(1, num_tensors, figsize=(num_tensors * 5, 5))

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
            ax.set_title("Image")
        ax.axis("off")  # Hide axes for better visualization

    plt.show()
