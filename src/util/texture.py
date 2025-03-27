from torch import Tensor
import PIL.Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def preprocess_image(image: PIL.Image) -> Tensor:
    """
    Preprocess image for neural style transfer.
    
    @param: image: Image to preprocess.
    
    :return: Preprocessed image tensor.
    """
    image = image.convert("L")  # Convert image to grayscale
    image = image.resize((256, 256))  # Resize image to 256x256
    return convert_image_to_tensor(image)

def convert_image_to_tensor(image: PIL.Image) -> Tensor:
    """
    Convert PIL image to tensor.
    
    @param: image: PIL image to convert to tensor.
    
    :return: Tensor of image.
    """
    return transform(image).unsqueeze(0)

def convert_tensor_to_image(tensor: Tensor) -> PIL.Image:
    """
    Convert tensor to PIL image.
    
    @param: tensor: Tensor to convert to PIL image.
    
    :return: PIL image of tensor.
    """
    transform = transforms.ToPILImage()
    return transform(tensor.squeeze(0))