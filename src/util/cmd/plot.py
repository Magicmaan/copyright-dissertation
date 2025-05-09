import os
from pathlib import Path
import argparse
from typing import TypedDict
import numpy as np

DATA_FILE = Path("analysis.txt")


class AnalysisResults(TypedDict):
    wm_vs_styled_wm_PSNR: float
    wm_vs_styled_wm_SSIM: float
    wm_vs_styled_wm_PIXEL: float
    wm_vs_styled_wm_PERCEPTUAL: float

    ext_vs_styled_ext_PSNR: float
    ext_vs_styled_ext_SSIM: float
    ext_vs_styled_ext_PIXEL: float
    ext_vs_styled_ext_PERCEPTUAL: float

    styled_vs_styled_wm_PSNR: float
    styled_vs_styled_wm_SSIM: float
    styled_vs_styled_wm_PIXEL: float
    styled_vs_styled_wm_PERCEPTUAL: float


TEXT_STRINGS = {
    "Pixel Difference (Watermarked vs Styled Watermarked)": "wm_vs_styled_wm_PIXEL",
    "Perceptual Difference (Watermarked vs Styled Watermarked)": "wm_vs_styled_wm_PERCEPTUAL",
    "PSNR (Watermarked vs Styled Watermarked)": "wm_vs_styled_wm_PSNR",
    "Structural Difference (Watermarked vs Styled Watermarked)": "wm_vs_styled_wm_SSIM",
    "Pixel Difference (Extracted vs Styled Extracted)": "ext_vs_styled_ext_PIXEL",
    "Perceptual Difference (Extracted vs Styled Extracted)": "ext_vs_styled_ext_PERCEPTUAL",
    "PSNR (Extracted vs Styled Extracted)": "ext_vs_styled_ext_PSNR",
    "Structural Difference (Extracted vs Styled Extracted)": "ext_vs_styled_ext_SSIM",
    "Pixel Difference (Styled vs Styled Watermarked)": "styled_vs_styled_wm_PIXEL",
    "Perceptual Difference (Styled vs Styled Watermarked)": "styled_vs_styled_wm_PERCEPTUAL",
    "PSNR (Styled vs Styled Watermarked)": "styled_vs_styled_wm_PSNR",
    "Structural Difference (Styled vs Styled Watermarked)": "styled_vs_styled_wm_SSIM",
}


def plot_analysis_results(analysis_results: list[AnalysisResults]):
    import matplotlib.pyplot as plt

    def plot_with_best_fit(
        x_values, y_values, x_label, y_label, title, colour, label, file_name
    ):
        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, y_values, c=colour, label=label)

        # Calculate and plot line of best fit
        if len(x_values) > 1 and len(y_values) > 1:
            coefficients = np.polyfit(x_values, y_values, 1)
            polynomial = np.poly1d(coefficients)
            best_fit_line = polynomial(x_values)
            plt.plot(
                x_values, best_fit_line, c="black", linestyle="--", label="Best Fit"
            )

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    # Pixel Difference vs PSNR (WM vs Styled WM)
    pixel_values = [result["wm_vs_styled_wm_PIXEL"] for result in analysis_results]
    psnr_values = [result["wm_vs_styled_wm_PSNR"] for result in analysis_results]
    plot_with_best_fit(
        pixel_values,
        psnr_values,
        "Pixel Difference",
        "PSNR",
        "Pixel Difference vs PSNR (WM vs Styled WM)",
        "blue",
        "WM vs Styled WM",
        "pixel_vs_psnr_wm_vs_styled_wm.png",
    )

    # Perceptual Difference vs SSIM (WM vs Styled WM)
    perceptual_values = [
        result["wm_vs_styled_wm_PERCEPTUAL"] for result in analysis_results
    ]
    ssim_values = [result["wm_vs_styled_wm_SSIM"] for result in analysis_results]
    plot_with_best_fit(
        perceptual_values,
        ssim_values,
        "Perceptual Difference",
        "SSIM",
        "Perceptual Difference vs SSIM (WM vs Styled WM)",
        "green",
        "WM vs Styled WM",
        "perceptual_vs_ssim_wm_vs_styled_wm.png",
    )

    # PSNR vs Perceptual Difference (WM vs Styled WM)
    psnr_values = [result["wm_vs_styled_wm_PSNR"] for result in analysis_results]
    perceptual_values = [
        result["wm_vs_styled_wm_PERCEPTUAL"] for result in analysis_results
    ]
    plot_with_best_fit(
        psnr_values,
        perceptual_values,
        "PSNR",
        "Perceptual Difference",
        "PSNR vs Perceptual Difference (WM vs Styled WM)",
        "red",
        "WM vs Styled WM",
        "psnr_vs_perceptual_wm_vs_styled_wm.png",
    )

    # Pixel Difference vs SSIM (WM vs Styled WM)
    pixel_values = [result["wm_vs_styled_wm_PIXEL"] for result in analysis_results]
    ssim_values = [result["wm_vs_styled_wm_SSIM"] for result in analysis_results]
    plot_with_best_fit(
        pixel_values,
        ssim_values,
        "Pixel Difference",
        "SSIM",
        "Pixel Difference vs SSIM (WM vs Styled WM)",
        "purple",
        "WM vs Styled WM",
        "pixel_vs_ssim_wm_vs_styled_wm.png",
    )

    # Pixel Difference vs PSNR (Extracted vs Styled Extracted)
    pixel_values = [result["ext_vs_styled_ext_PIXEL"] for result in analysis_results]
    psnr_values = [result["ext_vs_styled_ext_PSNR"] for result in analysis_results]
    plot_with_best_fit(
        pixel_values,
        psnr_values,
        "Pixel Difference",
        "PSNR",
        "Pixel Difference vs PSNR (Extracted vs Styled Extracted)",
        "orange",
        "Extracted vs Styled Extracted",
        "pixel_vs_psnr_ext_vs_styled_ext.png",
    )

    # Perceptual Difference vs SSIM (Extracted vs Styled Extracted)
    perceptual_values = [
        result["ext_vs_styled_ext_PERCEPTUAL"] for result in analysis_results
    ]
    ssim_values = [result["ext_vs_styled_ext_SSIM"] for result in analysis_results]
    plot_with_best_fit(
        perceptual_values,
        ssim_values,
        "Perceptual Difference",
        "SSIM",
        "Perceptual Difference vs SSIM (Extracted vs Styled Extracted)",
        "cyan",
        "Extracted vs Styled Extracted",
        "perceptual_vs_ssim_ext_vs_styled_ext.png",
    )

    # Pixel Difference vs PSNR (Styled vs Styled Watermarked)
    pixel_values = [result["styled_vs_styled_wm_PIXEL"] for result in analysis_results]
    psnr_values = [result["styled_vs_styled_wm_PSNR"] for result in analysis_results]
    plot_with_best_fit(
        pixel_values,
        psnr_values,
        "Pixel Difference",
        "PSNR",
        "Pixel Difference vs PSNR (Styled vs Styled Watermarked)",
        "brown",
        "Styled vs Styled Watermarked",
        "pixel_vs_psnr_styled_vs_styled_wm.png",
    )

    # Perceptual Difference vs SSIM (Styled vs Styled Watermarked)
    perceptual_values = [
        result["styled_vs_styled_wm_PERCEPTUAL"] for result in analysis_results
    ]
    ssim_values = [result["styled_vs_styled_wm_SSIM"] for result in analysis_results]
    plot_with_best_fit(
        perceptual_values,
        ssim_values,
        "Perceptual Difference",
        "SSIM",
        "Perceptual Difference vs SSIM (Styled vs Styled Watermarked)",
        "magenta",
        "Styled vs Styled Watermarked",
        "perceptual_vs_ssim_styled_vs_styled_wm.png",
    )


def main():
    parser = argparse.ArgumentParser(description="Plot analysis results.")
    parser.add_argument(
        "folder_directory",
        type=str,
        help="Path to the folder directory containing subfolders with analysis.txt files.",
    )
    args = parser.parse_args()

    folder_directory = Path(args.folder_directory)
    assert folder_directory.exists(), "The specified folder directory does not exist."
    assert folder_directory.is_dir(), "The specified path is not a directory."

    analysis_files = list(folder_directory.glob("**/analysis.txt"))

    assert (
        len(analysis_files) > 0
    ), "No analysis.txt files found in the specified directory."

    analysis_results: list[AnalysisResults] = []

    for analysis_file in analysis_files:
        assert analysis_file.exists(), f"File {f} does not exist."
        assert analysis_file.is_file(), f"{f} is not a file."

        with open(analysis_file, "r") as file:
            lines = file.readlines()
            results = AnalysisResults(
                wm_vs_styled_wm_PSNR=0.0,
                wm_vs_styled_wm_SSIM=0.0,
                wm_vs_styled_wm_PIXEL=0.0,
                wm_vs_styled_wm_PERCEPTUAL=0.0,
                ext_vs_styled_ext_PSNR=0.0,
                ext_vs_styled_ext_SSIM=0.0,
                ext_vs_styled_ext_PIXEL=0.0,
                ext_vs_styled_ext_PERCEPTUAL=0.0,
                styled_vs_styled_wm_PSNR=0.0,
                styled_vs_styled_wm_SSIM=0.0,
                styled_vs_styled_wm_PIXEL=0.0,
                styled_vs_styled_wm_PERCEPTUAL=0.0,
            )
            for line in lines:
                parts = line.strip().split(":")

                if len(parts) != 2:
                    continue

                if parts[0] in TEXT_STRINGS:
                    key = TEXT_STRINGS[parts[0]]
                    value = float(parts[1].strip())
                    results[key] = value
                # Process each line as needed
                print(line.strip())

            analysis_results.append(results)
            print(f"Processed {analysis_file}")

    print("Analysis Results:")
    for result in analysis_results:
        print(result)
    # print(f"Found {len(analysis_files)} analysis files.")

    plot_analysis_results(analysis_results)


if __name__ == "__main__":
    main()
