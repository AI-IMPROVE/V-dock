import torch
from torch import nn
from torch import optim
from utils import *


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_file = 'smiles_plus_features.csv'

    print("Loading data...")
    train_data, length = load_data(data_file, 'train_ind.npy', 'Docking_Score', 8192)
    val_data, length = load_data(data_file, 'val_ind.npy', 'Docking_Score', 8192)
    model = Net(length, 1, dropout=0.3).to(device=device)
    print(model)

    epochs = 20
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_loss = train_epoch(train_data, model, criterion, optimizer, device)

        if epoch % 10 == 0:
            val_true, val_pred, val_loss = eval_epoch(val_data, model, criterion, device)
            corr = pearson_r(val_true, val_pred)

            print(f'\nEpoch {epoch}')
            print('---------------------------------------')
            print(f'Train loss: {train_loss}')
            print(f'Validation loss: {val_loss}')
            print(f'Pearson R: {corr}')

    torch.save(model.state_dict(), 'l3_pytorch_model_multi_520.pt')


if __name__ == '__main__':
    main()
