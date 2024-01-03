import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from utils.data import desired_order


def process_batch(model, data, loss_fn, additional_features, optimizer=None):
    if additional_features:
        x, add_feat, y = data
    else:
        x, y = data
        add_feat = None
    if optimizer:
        optimizer.zero_grad()

    pred = model(x, add_feat) if add_feat else model(x)
    loss = loss_fn(pred, y)

    if optimizer:
        loss.backward()
        optimizer.step()

    return loss.item(), pred, y


def train_loop(dataloader, model, loss_fn, optimizer, additional_features: bool = False):
    size = len(dataloader.dataset)
    losses = []
    model.train()

    for batch, data in enumerate(dataloader):
        loss, _, _ = process_batch(model, data, loss_fn, additional_features, optimizer)
        losses.append(loss)

        if batch % 100 == 0:
            current = (batch + 1) * len(data[0])
            average_loss = sum(losses) / len(losses)
            print(f"loss: {average_loss:>7f}  [{current:>5d}/{size:>5d}]")

    return sum(losses) / len(losses)


def test_loop(dataloader, model, loss_fn, additional_features: bool = False):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    test_loss = []
    cm_pred = []
    cm_true = []

    with torch.no_grad():
        for data in dataloader:
            loss, pred, y = process_batch(model, data, loss_fn, additional_features)
            test_loss.append(loss)

            _p = pred.argmax(dim=1).cpu().numpy()
            _y = y.argmax(dim=1).cpu().numpy()
            correct += np.sum(_y == _p)
            cm_pred.extend(_p)
            cm_true.extend(_y)

            if not additional_features:
                plt.imshow(data[0][0].cpu().permute(1, 2, 0))
                plt.show()

    accuracy = correct / size
    average_loss = sum(test_loss) / len(test_loss)
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {average_loss:>8f} \n")

    cf_matrix = confusion_matrix(cm_true, cm_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in desired_order],
                         columns=[i for i in desired_order])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
