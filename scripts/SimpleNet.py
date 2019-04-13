from torchvision import models
from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=2)
        self.rrelu1 = nn.RReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(18, 18, kernel_size=5, stride=1, padding=2)
        # 4608 input features, 64 output features (see sizing flow below)
        self.fc1 = nn.Linear(18 * 8 * 8, 384)
        self.drop = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(384, 192)
        # 64 input features, 10 output features for our 10 defined classes
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        x = self.conv1(x)
        x = self.rrelu1(x)
        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.rrelu1(x)
        x = self.pool(x)
        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 8 * 8)
        # Computes the activation of the first fully connected layer
        x = self.fc1(x)
        x = self.rrelu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.rrelu1(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x