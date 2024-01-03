import torch.nn
import torch.nn.functional as F


class BarleyClassificationModel(torch.nn.Module):
    def __init__(self, img_size: tuple, additionalFeatures: bool = False):
        super(BarleyClassificationModel, self).__init__()
        self.additionalFeatures = additionalFeatures
        self.filters = [64, 128, 256, 256]
        features = 0

        # Wejście obrazy 170x80
        self.img_size = img_size
        if additionalFeatures:
            features = self.calc_size(len(self.filters)) + 75
        else:
            features = self.calc_size(len(self.filters))
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=self.filters[0], kernel_size=3, stride=1, padding=0)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=self.filters[0], out_channels=self.filters[1], kernel_size=3, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = torch.nn.Conv2d(in_channels=self.filters[1], out_channels=self.filters[2], kernel_size=3, stride=1, padding=0)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.conv4 = torch.nn.Conv2d(in_channels=self.filters[2], out_channels=self.filters[3], kernel_size=3, stride=1, padding=0)
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=(2, 2))
        if additionalFeatures:
            self.fc1 = torch.nn.Linear(in_features=features, out_features=128)
            self.bn = torch.nn.BatchNorm1d(num_features=128)
            self.fc2 = self.fc2 = torch.nn.Linear(in_features=128, out_features=5)
        else:
            self.fc1 = torch.nn.Linear(in_features=features, out_features=128)
            self.fc2 = torch.nn.Linear(in_features=128, out_features=5)
        self.drop = torch.nn.Dropout(p=0.2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.tensor, add_feat: torch.tensor=None):

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))

        x = torch.flatten(x, start_dim=1)
        if self.additionalFeatures:
            x = torch.cat((x, add_feat), dim=1)
            x = self.fc1(x)
            x = self.bn(x)
        else:
            x = self.fc1(x)
            x = self.drop(x)
        x = self.softmax(self.fc2(x))

        return x

    def calc_size(self, num_of_conv):
        for i in range(num_of_conv):
            self.img_size = tuple(x-2 for x in self.img_size)
            self.img_size = tuple(x // 2 for x in self.img_size)
        return self.img_size[0] * self.img_size[1] * self.filters[-1]


