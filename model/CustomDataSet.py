from pathlib import Path
import pandas as pd
import torchvision
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
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
        else:
            img = Path(self.features.iloc[item])
        img = Image.open(img)
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.PILToTensor(),
             torchvision.transforms.Resize(self.img_size, antialias=True),
             torchvision.transforms.ConvertImageDtype(torch.float32)])
        img = transform(img)
        target = self.target.iloc[item, :]
        return img.to(self.device), torch.tensor(target.values, dtype=torch.float32).to(self.device)


def prepareDataSet(features_csv: Path, img_csv: Path, img_size: tuple, additional_features_mod: bool = False):
    features_csv = pd.read_csv(features_csv, header=0)
    img_csv = pd.read_csv(img_csv, header=0)
    if additional_features_mod:
        x = img_csv.iloc[:, 1]
        feat = features_csv.iloc[:, :-1]
        x = pd.concat(x, feat, axis=1)
        y = features_csv.iloc[:, 75]
    else:
        x = img_csv.iloc[:, 1]
        y = features_csv.iloc[:, 75]

    y = pd.get_dummies(y)
    y = y[desired_order]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=10)

    train_ds = CustomImageDataset(x_train, y_train, img_size, additional_features_mod)
    test_ds = CustomImageDataset(x_test, y_train, img_size, additional_features_mod)

    return train_ds, test_ds

    pass
