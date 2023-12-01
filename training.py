import os
import candle
import torch
from torch import nn
from torch import optim
from utils import *

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')
n_gpu = 'cuda:' + os.environ['CUDA_VISIBLE_DEVICES']

additional_definitions = []

required = [
    'epochs',
    'batch_size',
    'learning_rate',
    'dropout',
    'output_dir'
]


class Vdock(candle.Benchmark):
    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions


def initialize_parameters():
    vdock_common = Vdock(file_path,
        'vdock_default_model.txt',
        'pytorch',
        prog='vdock_candle',
        desc='CANDLE compliant V-dock'
    )

    gParameters = candle.finalize_parameters(vdock_common)

    return gParameters


def run(gParameters):
    batch_size = gParameters['batch_size']
    epochs = gParameters['epochs']
    learning_rate = gParameters['learning_rate']
#    optimizer = gParameters['optimizer']
    loss = gParameters['loss']
    dropout = gParameters['dropout']
    output_dir = gParameters['output_dir']

    device = torch.device(n_gpu if torch.cuda.is_available() else 'cpu')
    data_file = 'smiles_plus_features.csv'

    print("Loading data...")
    train_data, length = load_data(data_file, 'train_ind.npy',
                                   'Docking_Score', batch_size)
    val_data, length = load_data(data_file, 'val_ind.npy',
                                 'Docking_Score', batch_size)
    model = Net(length, 1, dropout=dropout).to(device=device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = train_epoch(train_data, model, criterion,
                                 optimizer, device)

        if epoch % 5 == 0:
            val_true, val_pred, val_loss = eval_epoch(val_data, model,
                                                      criterion, device)
            corr = pearson_r(val_true, val_pred)

            print(f'\nEpoch {epoch}')
            print('---------------------------------------')
            print(f'Train loss: {train_loss}')
            print(f'Validation loss: {val_loss}')
            print(f'Pearson R: {corr}')

    torch.save(model.state_dict(), output_dir + '/model.hdf5')

    return history


def main():
    gParameters = initialize_parameters()
    history = run(gParameters)


if __name__ == '__main__':
    main()
