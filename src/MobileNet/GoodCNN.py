
# first attempt
# 94% validation accuracy
class CNNet(nn.Module):

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super(CNNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),
            nn.BatchNorm2d(12)
        )

        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Flatten(1),
            nn.Linear(144, 120),
            nn.Linear(120, 64),
            nn.Linear(64, self.num_classes),

        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
