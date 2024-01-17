from pathlib import Path

import torch
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.CustomDataSet import prepare_data_set
from model.model import BarleyClassificationModel
from utils.data import features_csv, img_csv
from utils.loops import test_loop, train_loop

if __name__ == '__main__':
    losses = []
    timestr = time.strftime("%d%m\\%H%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    additional_features = False

    train_ds, test_ds = prepare_data_set(features_csv, img_csv,
                                         (80, 170), additional_features_mod=additional_features)

    test_ds_loader = DataLoader(test_ds, batch_size=8, shuffle=True)
    train_ds_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    loss_fun = torch.nn.CrossEntropyLoss()
    model = BarleyClassificationModel((80, 170), add_feat=additional_features).to(device)
    optim = torch.optim.Adam(model.parameters(), 0.0001)
    epochs = 80

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        losses.append(train_loop(train_ds_loader, model, loss_fun, optim, add_feat=additional_features))
        if (t+1) % 20 == 0:
            test_loop(test_ds_loader, model, loss_fun, add_feat=additional_features)
    plt.plot(losses)
    plt.show()
    cnn_path = Path(f"Data\\CNN\\{timestr}")
    fc_path = Path(f"Data\\FC\\{timestr}")
    if not cnn_path.exists() and not fc_path.exists():
        Path.mkdir(cnn_path, parents=True)
        Path.mkdir(fc_path, parents=True)
    cnn_path = cnn_path.joinpath("model.pth")
    fc_path = fc_path.joinpath("model.pth")
    cnn_path = cnn_path.resolve()
    fc_path = fc_path.resolve()
    torch.save(model.backbone, cnn_path)
    torch.save(model.head, fc_path)
