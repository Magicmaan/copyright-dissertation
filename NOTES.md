
Despite Neural Style Transfer (NST) altering the pixel distribution of an image, high-level feature representations extracted from deep convolutional networks such as VGG19 retain structural traces of embedded watermarks. Therefore, the presence of a watermark can be detected from the difference in VGG19 feature activations between a watermarked and a stylized image.





converting a torch tensor to a numpy array and back erases the gradient and doesn't allow it to be tracked. now using pytorch wavelets
https://pytorch-wavelets.readthedocs.io/en/latest/dwt.html#differences-to-pywavelets 