import yaml
with open('config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

import torch
import torch.nn as nn
import torch.nn.functional as F
from bricks import Conv_IN_Act



class Discriminator(nn.Module):
    def __init__(self, inChannel=3, features=[64,128,256,512]):
        super().__init__()

        layers = [Conv_IN_Act(inChannel, features[0], stride=2, normalize=False)]
        
        inChannel = features[0]
        for feature in features[1:]:
            layers.append(Conv_IN_Act(inChannel, feature, stride=2, normalize=True))
            inChannel = feature
        
        # following origina code if layers > 4
        if features[-1] > 512:
            layers.append(Conv_IN_Act(features[-1], 512, stride=2, normalize=True))
        
        # squash output to one channel PatchGAN
        layers.append(nn.Conv2d(inChannel, 1, 4, stride=1, padding=1, padding_mode='reflect'))
        self.disc = nn.Sequential(*layers)
        self.init_weights()
        
    def forward(self, x):
        x = self.disc(x)
        return torch.sigmoid(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

# from torchsummary import summary
# x = torch.randn((5, 3, 256, 256))
# model = Discriminator()
# preds = model(x)
# summary(model, (3,256,256), depth=7)
# print(preds.shape)