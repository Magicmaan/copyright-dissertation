import torch
import torch.nn as nn

from util.dctdwt import embedWatermark


class Generator(nn.Module):
    """
    The NN generator model for the program.
    """

    def __init_(self):
        super(Generator, self).__init__()
        # Convolutional layers for embedding watermark in frequency domain
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # alphas for DCT DWT watermark embedding
        # these are the scaling factors for the watermark
        # used for the forward function

        # TODO: seperate alphas for each frequency band
        self.alphaDWTList = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(4)]
        )
        self.alphaDCT = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.1)) for _ in range(4)]
        )

        # use relu activation function
        # to ensure that the output is non-linear and can't be easily predicted

    # function to forward propagate the input
    def forward(
        self, image: torch.Tensor, watermark: torch.Tensor
    ) -> list[torch.Tensor, torch.Tensor]:
        """
        Forward propagate the input through the generator.
        :param: image: Content image tensor.
        :param: watermark: Watermark image tensor.

        :return: Output of the generator.
        """
        [watermarkedImage, extracted, _, _, _, _] = embedWatermark(
            image,
            watermark,
            alphaDWT=self.alphaDWTList,
            DCT_alpha=self.alphaDCT,
            display=False,
        )

        return [watermarkedImage, extracted]
