from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2, InterpolationMode

from utils.data import desired_order


class CustomImageDataset(Dataset):
    def __init__(self, features, target, img_size: tuple, additional_features_mod: bool):
        self.features = features
        self.target = target
        self.img_size = img_size
        self.additional_features_mod = additional_features_mod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        if self.additional_features_mod:
            img = Path(self.features.iloc[item, 0])
            a_feat = self.features.iloc[item, 1:]
            a_feat = a_feat.values.astype(np.float32)
        else:
            img = Path(self.features.iloc[item])
        img = self.__get_img(img)
        target = self.target.iloc[item, :]
        if self.additional_features_mod:
            return img.to(self.device), torch.tensor(a_feat, dtype=torch.float32).to(self.device), torch.tensor(
                target.values, dtype=torch.float32).to(self.device)
        else:
            return img.to(self.device), torch.tensor(target.values, dtype=torch.float32).to(self.device)

    def __get_img(self, img):
#        img2 = copy(img)
        img = Image.open(img)
        try:
            img2 = Image.open(img)
        except Exception:
            pass
        ratio = float(img.height) / img.width
        if ratio > 1:
            diff = img.height - img.width
            diff //= 2
            transform = v2.Compose(
                [v2.PILToTensor(),
                 #v2.Pad([diff, 0, diff, 0], fill=0),
                 v2.ConvertImageDtype(torch.float32),
                 v2.Resize(self.img_size, antialias=True),
                 #v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
                ])
        else:
            diff = img.width - img.height
            diff //= 2
            transform = v2.Compose(
                [v2.PILToTensor(),
                 #v2.Pad([0, diff, 0, diff], fill=0),
                 v2.ConvertImageDtype(torch.float32),
                 v2.Resize(self.img_size, antialias=True),
                 #v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR)
                 ])
        img = transform(img)
#        s = torch.permute(img.clone(), (1, 2, 0))
#        plt.imshow(s)
#        plt.show()
        return transform(img)


def prepareDataSet(features_csv: Path, img_csv: Path, img_size: tuple, additional_features_mod: bool = False):
    features_csv = pd.read_csv(features_csv, header=0)
    img_csv = pd.read_csv(img_csv, header=0)
    if additional_features_mod:
        x = img_csv.iloc[:, 1]
        feat = features_csv.iloc[:, :-1]
        x = pd.concat([x, feat], axis=1)
        y = features_csv.iloc[:, -1]
    else:
        x = img_csv.iloc[:, 1]
        y = features_csv.iloc[:, -1]

    y = pd.get_dummies(y)
    y = y[desired_order]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=11)

    train_ds = CustomImageDataset(x_train, y_train, img_size, additional_features_mod)
    test_ds = CustomImageDataset(x_test, y_test, img_size, additional_features_mod)

    return train_ds, test_ds

    pass
