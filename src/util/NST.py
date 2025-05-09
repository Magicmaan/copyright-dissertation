from pathlib import Path
import sys
import os
import subprocess
from torch import Tensor
import torch
import tempfile
from PIL import Image
from util.image import imageToTensor, tensorToImage
from typing import Literal
import shutil


# get link to my Neural Style Transfer script
# has to be added by user in nstPath.txt
ROOT_DIRECTORY = Path(__file__).resolve().parent.parent.parent
GATYS_PATH_FOLDER = ROOT_DIRECTORY / "GATYS_PATH.txt"
assert (
    GATYS_PATH_FOLDER.exists()
), f"GATYS_PATH.txt file not found at {GATYS_PATH_FOLDER}"
with GATYS_PATH_FOLDER.open("r") as file:
    GATYS_PATH = file.readline().strip()
assert GATYS_PATH, "NST_PATH.txt is empty or does not contain a valid path"
assert Path(
    GATYS_PATH
).exists(), f"The path in NST_PATH.txt does not exist: {GATYS_PATH}"
print("GATYS_PATH:", GATYS_PATH)

ADAIN_PATH_FOLDER = ROOT_DIRECTORY / "ADAIN_PATH.txt"
assert (
    ADAIN_PATH_FOLDER.exists()
), f"ADAIN_PATH.txt file not found at {ADAIN_PATH_FOLDER}"
with ADAIN_PATH_FOLDER.open("r") as file:
    ADAIN_PATH = file.readline().strip()
assert ADAIN_PATH, "ADAIN_PATH.txt is empty or does not contain a valid path"
assert Path(
    ADAIN_PATH
).exists(), f"The path in ADAIN_PATH.txt does not exist: {ADAIN_PATH}"
print("ADAIN_PATH:", ADAIN_PATH)

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

ModeType = Literal["gatys", "adain"]


def performNST(
    contentImage: Tensor,
    styleImage: Tensor,  # type: ignore[assignment]
    iterations: int = 101,
    lr: float = 0.05,
    alpha: float = 8,
    beta: float = 70,
    mode: ModeType = "gatys",
) -> Tensor:
    # print("path:", GATYS_PATH)

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
    # print("Content Image saved to:", tempContentPath)

    tempStylePath = ROOT_DIRECTORY / "temp" / "style.jpg"
    tempStylePath.parent.mkdir(parents=True, exist_ok=True)
    assert (
        tempStylePath.parent.exists()
    ), f"Directory does not exist: {tempStylePath.parent}"
    styleImagePIL = tensorToImage(styleImage)
    styleImagePIL.save(tempStylePath, format="JPEG")

    # print("Style Image saved to:", tempStylePath)

    print("Calling Neural Style Transfer Script...")

    match mode:
        case "gatys":
            run_gatys(
                temp_content_path=tempContentPath,
                temp_style_path=tempStylePath,
                output_path=outputPath,
                iterations=iterations,
                lr=lr,
                alpha=alpha,
                beta=beta,
            )

        case "adain":
            run_adain(
                temp_content_path=tempContentPath,
                temp_style_path=tempStylePath,
                output_path=outputPath,
            )

    finalOutputPath = outputPath / "final_output.jpg"
    assert finalOutputPath.exists(), f"Output image not found at {finalOutputPath}"
    result = Image.open(finalOutputPath)
    print("Neural Style Transfer Script finished.")
    return imageToTensor(result)


def run_gatys(
    temp_content_path: Path,
    temp_style_path: Path,
    output_path: Path,
    iterations: int,
    lr: float,
    alpha: float,
    beta: float,
) -> None:
    """
    Run the Neural Style Transfer (gatys) script with the given parameters.

    Args:
        nst_path: Path to the NST script.
        temp_content_path: Path to the temporary content image.
        temp_style_path: Path to the temporary style image.
        output_path: Path to save the output image.
        iterations: Number of iterations for the NST.
        lr: Learning rate for the optimizer.
        alpha: Weight for the content loss.
        beta: Weight for the style loss.
    """
    output_path = output_path.resolve()
    command = [
        "python",
        str(GATYS_PATH),
        str(temp_content_path.absolute()),
        str(temp_style_path.absolute()),
        str(output_path.absolute()),
    ]
    # Add model parameters
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

    # Run the command
    command = " ".join(command)
    # print("Command:", command)
    subprocess.run(command, shell=True, check=True)


# https://github.com/naoto0804/pytorch-AdaIN
def run_adain(
    temp_content_path: Path,
    temp_style_path: Path,
    output_path: Path,
) -> None:
    """
    Run the Neural Style Transfer (adain) script with the given parameters.

    Args:
        nst_path: Path to the NST script.
        temp_content_path: Path to the temporary content image.
        temp_style_path: Path to the temporary style image.
        output_path: Path to save the output image.
    """

    # CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg

    adain_output_path = Path(ADAIN_PATH) / "output"
    # Remove the existing output directory if it exists
    # just a jank because im lazy
    if adain_output_path.exists():
        for file in adain_output_path.iterdir():
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
        adain_output_path.rmdir()
    adain_output_path.mkdir(parents=True, exist_ok=True)

    output_path = output_path.resolve()
    command = [
        "python",
        str(ADAIN_PATH + "/test.py"),
        "--content",
        str(temp_content_path.absolute()),
        "--style",
        str(temp_style_path.absolute()),
    ]

    # Run the command
    command = " ".join(command)
    # print("Command:", command)
    subprocess.run(command, cwd=ADAIN_PATH, shell=True, check=True)

    # Move the output to the desired location
    # Find the output file in the adain_output_path
    output_file = next(adain_output_path.glob("*"), None)
    assert output_file is not None, f"No output file found in {adain_output_path}"

    # Move the output file to the desired location
    shutil.move(str(output_file), str(output_path / "final_output.jpg"))
