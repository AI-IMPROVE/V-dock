import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import pandas as pd


class Net(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.3):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Linear1 = nn.Linear(self.input_size, 1024)
        self.Linear2 = nn.Linear(1024, 528)
        self.Linear3 = nn.Linear(528, self.output_size)
        self.Batch1 = nn.BatchNorm1d(self.input_size)
        self.Drop = nn.Dropout(p=dropout)

    def forward(self, input_tensor):
        x = self.Batch1(input_tensor)
        x = self.Drop(x)
        x = F.elu(self.Linear1(x))
        x = F.elu(self.Linear2(x))
        x = self.Linear3(x)
        return x


def load_data(data_file, split, y_col, batch_size):
    X = pd.read_csv(data_file, index_col=0)
    y = X[y_col].to_numpy(dtype=np.float32).reshape(-1,1)
    X = X.drop(columns=[y_col]).to_numpy(dtype=np.float32)
    split_ind = np.load(split)

    X = torch.tensor(X[split_ind])
    y = torch.tensor(y[split_ind])
    df = TensorDataset(X, y)
    dataloader = DataLoader(df, batch_size=batch_size, shuffle=True)

    length = X.size(1)

    return dataloader, length


def train_epoch(data, model, criterion, optimizer, device):
    running_loss = 0.0
    epoch_step = 0
    model.train()

    for i, (xx,yy) in enumerate(data):
        xx = xx.to(device=device)
        yy = yy.to(device=device)

        output = model(xx)
        loss = criterion(output, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_step += 1

    return running_loss/epoch_step


def eval_epoch(data, model, criterion, device):
    running_loss = 0.0
    epoch_step = 0
    true = []
    pred = []

    model.eval()

    for i, (xx,yy) in enumerate(data):
        with torch.no_grad():
            xx = xx.to(device=device)
            yy = yy.to(device=device)
            output = model(xx)
            loss = criterion(output, yy)
            running_loss += loss.item()
            epoch_step += 1

            for i,x in enumerate(output):
                x = x.item()
                pred.append(x)

            for a,b in enumerate(yy):
                b = b.item()
                true.append(b)

    return true, pred, running_loss/epoch_step


def pearson_r(true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    vx = true - np.mean(true)
    vy = pred - np.mean(pred)
    corr = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))

    return corr
