import torch.nn as nn

# create decoder block inherited from nn.Module
class DecoderBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        
        # 1x1 projection module to reduce channels
        self.proj = nn.Sequential(
            # convolution
            nn.Conv2d(channels_in, channels_in // 4, kernel_size=1, bias=False),
            # batch normalization
            nn.BatchNorm2d(channels_in // 4),
            # relu activation
            nn.ReLU()
        )

        # fully convolutional module
        self.deconv = nn.Sequential(
            # deconvolution
            nn.ConvTranspose2d(
                channels_in // 4,
                channels_in // 4,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                groups=channels_in // 4,
                bias=False
            ),
            # batch normalization
            nn.BatchNorm2d(channels_in // 4),
            # relu activation
            nn.ReLU()
        )

        # 1x1 unprojection module to increase channels
        self.unproj = nn.Sequential(
            # convolution
            nn.Conv2d(channels_in // 4, channels_out, kernel_size=1, bias=False),
            # batch normalization
            nn.BatchNorm2d(channels_out),
            # relu activation
            nn.ReLU()
        )

    # stack layers and perform a forward pass
    def forward(self, x):

        proj = self.proj(x)
        deconv = self.deconv(proj)
        unproj = self.unproj(deconv)

        return unproj