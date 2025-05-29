import torch.nn as nn

class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 256)
        )

    def forward(self, x):
        return self.net(x)
