import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

from model.model import BarleyClassificationModel
from utils.data import desired_order


def process_batch(model: BarleyClassificationModel, data, loss_fn, add_feat, optimizer=None):
    if add_feat:
        x, add_feat, y = data
    else:
        x, y = data
        add_feat = None
    if optimizer:
        optimizer.zero_grad()

    pred = model(x, add_feat) if add_feat is not None else model(x)
    loss = loss_fn(pred, y)

    if optimizer:
        loss.backward()
        optimizer.step()

    return loss.item(), pred, y


def train_loop(dataloader, model: BarleyClassificationModel, loss_fn, optimizer, add_feat: bool = False):
    size = len(dataloader.dataset)
    losses = []
    model.train()
    if add_feat:
        model.freeze_cnn()

    for batch, data in enumerate(dataloader):
        loss, _, _ = process_batch(model, data, loss_fn, add_feat, optimizer)
        losses.append(loss)

        if batch % 100 == 0:
            current = (batch + 1) * len(data[0])
            average_loss = sum(losses) / len(losses)
            print(f"loss: {average_loss:>7f}  [{current:>5d}/{size:>5d}]")

    return sum(losses) / len(losses)


def test_loop(dataloader, model: BarleyClassificationModel, loss_fn, add_feat: bool = False):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    test_loss = []
    cm_pred = []
    cm_true = []

    with torch.no_grad():
        for data in dataloader:
            loss, pred, y = process_batch(model, data, loss_fn, add_feat)
            test_loss.append(loss)

            _p = pred.argmax(dim=1).cpu().numpy()
            _y = y.argmax(dim=1).cpu().numpy()
            correct += np.sum(_y == _p)
            cm_pred.extend(_p)
            cm_true.extend(_y)

    accuracy = correct / size
    average_loss = sum(test_loss) / len(test_loss)
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {average_loss:>8f} \n")

    cf_matrix = confusion_matrix(cm_true, cm_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in desired_order],
                         columns=[i for i in desired_order])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
