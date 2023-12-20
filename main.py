import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.CustomDataSet import prepareDataSet
from model.model import BarleyClassificationModel
from utils.data import features_csv, img_csv
from utils.loops import test_loop, train_loop

if __name__ == '__main__':
    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    additional_features = False

    train_ds, test_ds = prepareDataSet(features_csv, img_csv,
                                       (170, 170), additional_features_mod=additional_features)

    test_ds_loader = DataLoader(test_ds, batch_size=16, shuffle=True)
    train_ds_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    loss_fun = torch.nn.BCELoss()
    model = BarleyClassificationModel((170, 170), additionalFeatures=additional_features).to(device)
    optim = torch.optim.Adam(model.parameters(), 0.0001)
    epochs = 150

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        losses.append(train_loop(train_ds_loader, model, loss_fun, optim, additionalFeatures=additional_features))
        if ((t+1) % 10 == 0):
            test_loop(test_ds_loader, model, loss_fun, additionalFeatures=additional_features)
            plt.plot(losses)
            plt.show()
