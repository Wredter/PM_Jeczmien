from pathlib import Path

import torch
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.CustomDataSet import prepare_data_set
from model.model import BarleyClassificationModel
from utils.data import features_csv, img_csv
from utils.loops import test_loop, train_loop


losses = []
timestr = time.strftime("%m%d/%H%M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
additional_features = True
just_test = False
train_ds, test_ds = prepare_data_set(features_csv, img_csv,
                                     (80, 170), additional_features_mod=additional_features)

test_ds_loader = DataLoader(test_ds, batch_size=8, shuffle=True)
train_ds_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
loss_fun = torch.nn.BCELoss()
model = BarleyClassificationModel((80, 170), add_feat=additional_features).to(device)
optim = torch.optim.Adam(model.parameters(), 0.0001)
epochs = 75

cnn_path = Path("C:\\Users\\wikto\\PycharmProjects\\PM_Jeczmien\\Data\\CNN\\1201\\1853\\model.pth")
fc_path = Path("C:\\Users\\wikto\\PycharmProjects\\PM_Jeczmien\\Data\\FC\\1201\\1853\\model.pth")


if not just_test:
    model.backbone = torch.load(cnn_path)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        losses.append(train_loop(train_ds_loader, model, loss_fun, optim, add_feat=additional_features))
        if (t + 1) % 25 == 0:
            test_loop(test_ds_loader, model, loss_fun, add_feat=additional_features)
    plt.plot(losses)
    plt.show()
else:
    model.backbone = torch.load(cnn_path)
    model.head = torch.load(fc_path)
    test_loop(test_ds_loader, model, loss_fun, add_feat=additional_features)