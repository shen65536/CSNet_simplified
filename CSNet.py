import torch
import torch.nn as nn


class CSNet(nn.Module):
    def __init__(self, channels=1, ratio=0.15, block_size=32):
        super().__init__()
        measurement = int(block_size ** 2 * ratio)

        self.sample = nn.Conv2d(channels, measurement, kernel_size=32, padding=0, stride=32, bias=False)
        self.init = nn.Conv2d(measurement, channels * 32 ** 2, kernel_size=1, padding=0, stride=1, bias=False)

        self.conv1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(1, channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(32)
        self.relu = nn.ReLU(inplace=True)

        self.save_path = "./CSNet.pth"

    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        output = self.sample(x)
        output = self.init(output)
        output = self.pixel_shuffle(output)

        output = self.relu(self.conv1(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv2(output))
        output = self.conv3(output)
        return output
