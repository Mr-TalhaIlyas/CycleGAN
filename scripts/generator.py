import torch
import torch.nn as nn
from bricks import ResidualBlock, TransConvBlock, Conv_IN_Act


class Generator(nn.Module):
    def __init__(self, inChannel, genChannel=64, residual_blocks=9):
        super().__init__()
        self.stem  = Conv_IN_Act(inChannel, genChannel, kernel_size=7, stride=1, padding=3)

        self.down_blocks = nn.Sequential(
            Conv_IN_Act(genChannel, genChannel*2, kernel_size=3, stride=2, padding=1),
            Conv_IN_Act(genChannel*2, genChannel*4, use_act=False, kernel_size=3, stride=2, padding=1)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(genChannel*4) for _ in range(residual_blocks)]
        )

        self.up_blocks = nn.Sequential(
            TransConvBlock(genChannel*4, genChannel*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            TransConvBlock(genChannel*2, genChannel,   kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        self.out = nn.Conv2d(genChannel, inChannel, kernel_size=7, stride=1, padding=3, padding_mode='reflect')
        self.init_weights()
        
    def forward(self, x):

        x = self.stem(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        x = self.out(x)

        return torch.tanh(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

# from torchsummary import summary
# x = torch.randn((5, 3, 256, 256))
# model = Generator(3,64)
# preds = model(x)
# summary(model, (3,256,256), depth=7)
# print(preds.shape)