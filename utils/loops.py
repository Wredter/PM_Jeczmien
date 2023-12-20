import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from utils.data import desired_order


def train_loop(dataloader, model, loss_fn, optimizer, additionalFeatures: bool=False):
    size = len(dataloader.dataset)
    losses = []
    acc = []
    model.train()
    if additionalFeatures:
        for batch, (X, add_feat, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X, add_feat)
            loss = loss_fn(pred, y)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    else:
        for batch, (X, y) in enumerate(dataloader):

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)
            losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return sum(losses) / len(losses)


def test_loop(dataloader, model, loss_fn, additionalFeatures: bool=False):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    test_loss = []
    cm_pred = []
    cm_true = []

    with torch.no_grad():
        if additionalFeatures:
            for X, add_feat, y in dataloader:
                pred = model(X, add_feat)
                test_loss.append(loss_fn(pred, y).item())
                _p = pred.argmax(dim=1).cpu().numpy()
                _y = y.argmax(dim=1).cpu().numpy()
                correct += np.sum(_y == _p)
                cm_pred.extend(_p)
                cm_true.extend(_y)
        else:
            for X, y in dataloader:
                pred = model(X)
                test_loss.append(loss_fn(pred, y).item())
                _p = pred.argmax(dim=1).cpu().numpy()
                _y = y.argmax(dim=1).cpu().numpy()
                correct += np.sum(_y == _p)
                cm_pred.extend(_p)
                cm_true.extend(_y)
                plt.imshow(X[0].cpu().permute(1, 2, 0))

    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {sum(test_loss)/len(test_loss):>8f} \n")

    cf_matrix = confusion_matrix(cm_true, cm_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in desired_order],
                         columns=[i for i in desired_order])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
