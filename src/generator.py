import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    The NN generator model for the program.
    """

    def __init_(self):
        super(Generator, self).__init__()

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        # use relu activation function
        # to ensure that the output is non-linear and can't be easily predicted
        self.relu: nn.ReLU = nn.ReLU()

    # function to forward propagate the input
    def forward(self, content: torch.Tensor, watermark: any) -> torch.Tensor:
        """
        Forward propagate the input through the generator.
        :param: content: Content image.
        :param: watermark: Watermark image.

        :return: Output of the generator
        """
        content = self.relu(self.conv1(content))
        content = self.conv2(content)
        return content + watermark
