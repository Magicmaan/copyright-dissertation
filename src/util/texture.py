from torch import Tensor
import PIL.Image
from torchvision import transforms


# Define two transforms - one for grayscale and one for color
transform_gray = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
)

transform_color = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


def preprocessImage(image: PIL.Image, preserve_color: bool = True) -> Tensor:
    """
    Preprocess image for neural style transfer.

    :param: image: Image to preprocess.
    :param: preserve_color: If True, preserves color information.

    :return: Preprocessed image tensor.
    """
    if preserve_color:
        image = image.resize((256, 256))
        return transform_color(image).unsqueeze(0)
    else:
        # Convert to grayscale
        image = image.convert("L")
        image = image.resize((256, 256))
        return transform_gray(image).unsqueeze(0)


def imageToTensor(image: PIL.Image, preserve_color: bool = True) -> Tensor:
    """
    Convert PIL image to tensor.

    :param: image: PIL image to convert to tensor.
    :param: preserve_color: If True, preserves color information.

    :return: Tensor of image.
    """
    if preserve_color:
        return transform_color(image).unsqueeze(0)
    else:
        return transform_gray(image).unsqueeze(0)


def tensorToImage(tensor: Tensor) -> PIL.Image:
    """
    Convert tensor to PIL image.

    :param: tensor: Tensor to convert to PIL image.

    :return: PIL image of tensor.
    """
    # Make sure tensor has the right shape before converting
    if tensor.dim() == 4:  # If tensor has batch dimension
        tensor = tensor.squeeze(0)

    mode = "L"
    try:
        # Determine mode based on the number of channels
        if tensor.shape[0] == 1:
            mode = "L"  # Grayscale
        else:
            mode = "RGB"  # Color
    except IndexError:
        print("bruh")

    transform_to_pil = transforms.ToPILImage(mode=mode)
    return transform_to_pil(tensor)
