import torch.nn
import torch.nn.functional as F


class BarleyClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Wejście obrazy 170x80
        self.img_w = 80
        self.img_h = 170
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.fc1 = torch.nn.Linear(in_features=9728, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=5)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.tensor):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.softmax(self.fc2(x))

        return x
