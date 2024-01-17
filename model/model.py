import torch.nn


class BarleyClassificationModel(torch.nn.Module):
    def __init__(self, img_size: tuple, add_feat: bool = False):
        super(BarleyClassificationModel, self).__init__()
        self.add_feat = add_feat
        self.filters = [64, 128, 256, 256]

        # Wejście obrazy 170x80
        self.img_size = img_size
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,
                            out_channels=self.filters[0],
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(in_channels=self.filters[0],
                            out_channels=self.filters[1],
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(in_channels=self.filters[1],
                            out_channels=self.filters[2],
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(in_channels=self.filters[2],
                            out_channels=self.filters[3],
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(2, 2))
        )
        common_layers = []

        if add_feat:
            features = self.calc_size(len(self.filters)) + 75
            common_layers.append(torch.nn.BatchNorm1d(features))
        else:
            features = self.calc_size(len(self.filters))

        common_layers.extend([
            torch.nn.Linear(in_features=features, out_features=128),
            torch.nn.Tanh(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=128, out_features=5),
            torch.nn.Softmax(dim=1)])

        self.head = torch.nn.Sequential(*common_layers)

    def forward(self, x: torch.tensor, add_feat: torch.tensor = None):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        if self.add_feat:
            x = torch.cat((x, add_feat), dim=1)
            x = self.head(x)
        else:
            x = self.head(x)
        return x

    def calc_size(self, num_of_conv):
        for i in range(num_of_conv):
            self.img_size = tuple(x - 2 for x in self.img_size)
            self.img_size = tuple(x // 2 for x in self.img_size)
        return self.img_size[0] * self.img_size[1] * self.filters[-1]

    def freeze_cnn(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
