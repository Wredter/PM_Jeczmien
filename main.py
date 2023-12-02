import torch
from torch.utils.data import DataLoader

from model.CustomDataSet import prepareDataSet
from model.model import BarleyClassificationModel
from utils.data import features_csv, img_csv
from utils.loops import test_loop, train_loop

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, test_ds = prepareDataSet(features_csv, img_csv, (170, 80))

    test_ds_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    train_ds_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    loss_fun = torch.nn.CrossEntropyLoss()
    model = BarleyClassificationModel().to(device)
    optim = torch.optim.Adam(model.parameters(), 0.001)
    epochs = 150

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_ds_loader, model, loss_fun, optim)
        test_loop(test_ds_loader, model, loss_fun)
