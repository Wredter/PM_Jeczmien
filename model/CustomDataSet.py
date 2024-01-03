from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import Compose, PILToTensor, ConvertImageDtype, Resize
from utils.data import desired_order


class CustomImageDataset(Dataset):
    def __init__(self, features, target, img_size: tuple, additional_features_mod: bool):
        self.features = features
        self.target = target
        self.img_size = img_size
        self.additional_features_mod = additional_features_mod
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = Compose([
            PILToTensor(),
            ConvertImageDtype(torch.float32),
            Resize(self.img_size, antialias=True),
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        if self.additional_features_mod:
            img_path = Path(self.features.iloc[item, 0]).resolve()
            add_feat = self.features.iloc[item, 1:].values.astype(np.float32)
        else:
            img_path = Path(self.features.iloc[item]).resolve()

        img = self.__get_img(img_path)
        target = torch.tensor(self.target.iloc[item, :].values, dtype=torch.float32).to(self.device)

        if self.additional_features_mod:
            return img, torch.tensor(add_feat, dtype=torch.float32).to(self.device), target
        else:
            return img, target

    def __get_img(self, img_path):
        img = Image.open(img_path)
        return self.transform(img).to(self.device)


def prepare_data_set(features_csv: Path, img_csv: Path, img_size: tuple, additional_features_mod: bool = False):
    features_df = pd.read_csv(features_csv, header=0)
    img_df = pd.read_csv(img_csv, header=0)

    x = img_df.iloc[:, 1]
    if additional_features_mod:
        x = pd.concat([x, features_df.iloc[:, :-1]], axis=1)

    y = pd.get_dummies(features_df.iloc[:, -1])
    y = y[desired_order]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=11)

    train_ds = CustomImageDataset(x_train, y_train, img_size, additional_features_mod)
    test_ds = CustomImageDataset(x_test, y_test, img_size, additional_features_mod)

    return train_ds, test_ds
