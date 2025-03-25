from torch import Tensor
from PIL import Image
from torchvision import transforms


def preprocess_image(image: Image) -> Tensor:
    """
    Preprocess image for neural style transfer.
    
    @param: image: Image to preprocess.
    
    :return: Preprocessed image tensor.
    """

def convert_image_to_tensor(image: Image) -> Tensor:
    """
    Convert PIL image to tensor.
    
    @param: image: PIL image to convert to tensor.
    
    :return: Tensor of image.
    """
    return transforms.to_tensor(image)

def convert_tensor_to_image(tensor: Tensor) -> Image:
    """
    Convert tensor to PIL image.
    
    @param: tensor: Tensor to convert to PIL image.
    
    :return: PIL image of tensor.
    """
    return transforms.ToPILImage()(tensor)