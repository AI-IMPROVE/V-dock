import os
import candle
import torch
from torch import nn
from utils import *

file_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.environ['CANDLE_DATA_DIR'].rstrip('/')
n_gpu = 'cuda:' + os.environ['CUDA_VISIBLE_DEVICES']

additional_definitions = []

required = [
    'epochs',
    'batch_size',
    'learning_rate',
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
#    loss = gParameters['loss']
    output_dir = gParameters['output_dir']

    device = torch.device(n_gpu if torch.cuda.is_available() else 'cpu')
    data_file = data_dir + '/smiles_plus_features.csv'

    print("Loading data...")
    test_data, length = load_data(data_file, 'test_ind.npy',
                                  'Docking_Score', batch_size)
    model = Net(length, 1, dropout=0.3).to(device=device)

    criterion = nn.MSELoss()
    model.load_state_dict(torch.load(output_dir + '/model.hdf5'))
    model.eval()

    true, pred, test_loss = eval_epoch(test_data, model, criterion, device)

    return history


def main():
    gParameters = initialize_parameters()
    history = run(gParameters)


if __name__ == '__main__':
    main()
