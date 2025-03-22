import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 4, stride, 1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channel, features[0], 4, 2, 1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
        )
        layers = []
        in_channel = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channel, feature, stride=1 if feature == features[-1] else 2))
            in_channel = feature
        layers.append(nn.Conv2d(in_channel, 1, 4, 1, 1, padding_mode='reflect'))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.initial(x)
        x = self.model(x)
        return torch.sigmoid(x)


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 64, 64).to(device)
    model = Discriminator(in_channel=3, features=[64, 128, 256, 512]).to(device)
    y = model(x)
    print(y.size())

if __name__ == '__main__':
    test()