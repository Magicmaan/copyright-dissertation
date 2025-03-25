
class TotalLoss(nn.Module):
    """
    Calculate total loss for the program.
    
    @param: perceptual_weight: Weight for perceptual loss. ( )
    @param: watermark_weight: Weight for watermark loss.
    @param: adversarial_loss: Weight for adversarial loss.
    """
    
    def __init__(self, perceptual_weight, watermark_weight, adversarial_loss):
        super(TotalLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.watermark_weight = watermark_weight
        self.adversarial_loss = adversarial_loss

    def forward(self, perceptual, watermark, adversarial):
        return (
            self.perceptual_weight * perceptual + 
            self.watermark_weight * watermark + # may have to be minus? 
            self.adversarial_loss * adversarial
        )