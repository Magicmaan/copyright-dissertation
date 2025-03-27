from torch import Tensor
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

from util.texture import convert_tensor_to_image

def display_image_tensors(*tensors: Tensor):
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
        ax.imshow(image)
        ax.axis('off')  # Hide axes for better visualization
    
    plt.show()