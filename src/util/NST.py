from pathlib import Path
import sys
import os
import subprocess
from torch import Tensor
import torch
import tempfile
from PIL import Image
from util.image import imageToTensor, tensorToImage


# get link to my Neural Style Transfer script
# has to be added by user in nstPath.txt
ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
NST_PATH_FOLDER = ROOT_DIRECTORY / "NST_PATH.txt"
print("ROOT_DIRECTORY:", ROOT_DIRECTORY)
print("NST_PATH_FILE:", NST_PATH_FOLDER)
assert NST_PATH_FOLDER.exists(), f"NST_PATH.txt file not found at {NST_PATH_FOLDER}"
with NST_PATH_FOLDER.open("r") as file:
    NST_PATH = file.readline().strip()
assert NST_PATH, "NST_PATH.txt is empty or does not contain a valid path"
assert Path(NST_PATH).exists(), f"The path in NST_PATH.txt does not exist: {NST_PATH}"
print("NST_PATH:", NST_PATH)

# # Add the NST_PATH to the system path to import the script
# sys.path.append(NST_PATH)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# (
#     contentImage: Tensor,
#     styleImage: Tensor,
#     outputImage: Tensor,
#     outputPath: Union[str, Path],
#     iterations: int = 1001,
#     lr: float = 0.05,
#     alpha: float = 8,
#     beta: float = 70,
# ) -> Tensor:
def performNST(
    contentImage: Tensor,
    styleImage: Tensor,  # type: ignore[assignment]
    iterations: int = 101,
    lr: float = 0.05,
    alpha: float = 8,
    beta: float = 70,
) -> Tensor:
    print("path:", NST_PATH)

    outputPath = ROOT_DIRECTORY / "temp"

    # save tensors to image files temporarily, to be passed into NST
    tempContentPath = ROOT_DIRECTORY / "temp" / "content.jpg"
    tempContentPath.parent.mkdir(parents=True, exist_ok=True)
    tempContentPath.parent.mkdir(parents=True, exist_ok=True)
    assert (
        tempContentPath.parent.exists()
    ), f"Directory does not exist: {tempContentPath.parent}"
    contentImagePIL: Tensor = tensorToImage(contentImage)
    contentImagePIL.save(tempContentPath, format="JPEG")
    print("Content Image saved to:", tempContentPath)

    tempStylePath = ROOT_DIRECTORY / "temp" / "style.jpg"
    tempStylePath.parent.mkdir(parents=True, exist_ok=True)
    assert (
        tempStylePath.parent.exists()
    ), f"Directory does not exist: {tempStylePath.parent}"
    styleImagePIL = tensorToImage(styleImage)
    styleImagePIL.save(tempStylePath, format="JPEG")

    print("Style Image saved to:", tempStylePath)

    print("Calling Neural Style Transfer Script...")

    # perform NST. data is saved to root/temp
    outputPath = outputPath.resolve()
    command = [
        "python",
        str(NST_PATH),
        str(tempContentPath.absolute()),
        str(tempStylePath.absolute()),
        str(outputPath.absolute()),
    ]
    command += [
        "--iterations",
        str(iterations),
        "--lr",
        str(lr),
        "--alpha",
        str(alpha),
        "--beta",
        str(beta),
    ]
    command = " ".join(command)
    print("Command:", command)
    subprocess.run(command, shell=True, check=True)

    finalOutputPath = outputPath / "final_output.jpg"
    assert finalOutputPath.exists(), f"Output image not found at {finalOutputPath}"
    result = Image.open(finalOutputPath)
    print("Neural Style Transfer Script finished.")
    return imageToTensor(result)
