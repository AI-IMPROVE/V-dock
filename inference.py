import torch
from torch import nn
from utils import *


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_file = 'smiles_plus_features.csv'

    print("Loading data...")
    test_data, length = load_data(data_file, 'test_ind.npy', 'Docking_Score', 8192)
    model = Net(length, 1, dropout=0.3).to(device=device)

    criterion = nn.MSELoss()
    model.load_state_dict(torch.load('l3_pytorch_model_multi_520.pt'))
    model.eval()

    true, pred, test_loss = eval_epoch(test_data, model, criterion, device)


if __name__ == '__main__':
    main()
