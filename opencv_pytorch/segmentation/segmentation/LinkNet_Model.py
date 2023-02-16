import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.nn as nn
from segmentation.DecoderBlock import DecoderBlock

# create LinkNet model with ResNet18 encoder
class LinkNet(nn.Module):
    def __init__(self, num_classes, encoder="resnet18"):
        super().__init__()
        assert hasattr(models, encoder), "Undefined encoder type"
        # prepare feature extractor from `torchvision` ResNet model
        feature_extractor = getattr(models, encoder)(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Init block: get configured Conv2d, BatchNorm2d layers and ReLU from torch ResNet class
        self.init = nn.Sequential(feature_extractor.conv1, feature_extractor.bn1, feature_extractor.relu)
        self.maxpool = feature_extractor.maxpool
#         print(self.maxpool)

        # Encoder's blocks: torch ResNet18 blocks initialization
        self.layer1 = feature_extractor.layer1
        self.layer2 = feature_extractor.layer2
        self.layer3 = feature_extractor.layer3
        self.layer4 = feature_extractor.layer4

        # Decoder's block: DecoderBlock module
        self.up4 = DecoderBlock(self._num_channels(self.layer4), self._num_channels(self.layer3))
        self.up3 = DecoderBlock(self._num_channels(self.layer3), self._num_channels(self.layer2))
        self.up2 = DecoderBlock(self._num_channels(self.layer2), self._num_channels(self.layer1))
        self.up1 = DecoderBlock(self._num_channels(self.layer1), self._num_channels(self.layer1))

        # Classification block: define a classifier module
        self.classifier = nn.Sequential(
            # deconvolution layer
            nn.ConvTranspose2d(self._num_channels(self.layer1), 32, 3, stride=2, bias=False),
            # batch normalization with num_features = 32
            nn.BatchNorm2d(32),
            # activation function
            nn.ReLU(),
            # convolutional layer
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            # batch normalization with num_features = 32
            nn.BatchNorm2d(32),
            # activation function
            nn.ReLU(),
            # convolutional layer
            nn.Conv2d(32, num_classes, kernel_size=2, padding=0)
        )

    # get a compatible number of channels to stack all of the LinkNet's blocks together
    @staticmethod
    def _num_channels(block):
        """
           Extract batch-norm num_features from the input block.

            Arguments:
                block: torch resNet18 layers.
        """
        # check whether the input block is models.resnet.BasicBlock type
        if isinstance(block[-1], models.resnet.BasicBlock):
            return block[-1].bn2.weight.size(0)
        # if not extract the spatial characteristic of batch-norm weights from input block
        return block[-1].bn3.weight.size(0)

    # define the forward pass
    def forward(self, x):
        
        # output size = (64, 160, 160)
        init = self.init(x)
        
        # output size = (64, 80, 80)
        maxpool = self.maxpool(init)
        
        # output size = (64, 80, 80)
        layer1 = self.layer1(maxpool)
        
        # output size = (128, 40, 40)
        layer2 = self.layer2(layer1)
        
        # output size = (256, 20, 20)
        layer3 = self.layer3(layer2)
        
        # output size = (512, 10, 10)
        layer4 = self.layer4(layer3)
        
        # output size = (256, 20, 20)
        up4 = self.up4(layer4) + layer3
        
        # output size = (128, 40, 40)
        up3 = self.up3(up4) + layer2
        
        # output size = (64, 80, 80)
        up2 = self.up2(up3) + layer1
        
        # output size = (64, 160, 160)
        up1 = self.up1(up2)
        
        # output size = (5, 320, 320), where 5 is the predefined number of classes
        output = self.classifier(up1)
        

        return output