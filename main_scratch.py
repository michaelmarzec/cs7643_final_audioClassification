import numpy as np

from torch.utils.data import DataLoader

from models import LinearModel
from utils import utils
from utils import dataloader
import torch
from torch import nn
from torch import optim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)

    training_data = utils.load_pytorch_tensor('./utils/balanced_train_data.pt')
    training_label = utils.load_pytorch_tensor('./utils/balanced_train_label.pt')

    training_data = np.float32(training_data)
    training_label = np.float32(training_label)


    dataset = dataloader.MusicDataset(training_data, training_label)

    # The Dataloader class handles all the shuffles for you
    loader = iter(DataLoader(dataset, batch_size=32, shuffle=True))

    # for the linear model, the input will be all 10seconds of audio
    # stacked into one layer - therefore input dimension
    # is 10 * 128
    linear_model = LinearModel.LinearModel(10*128, 64, 527)
    optimizer = optim.Adam(linear_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    utils.train(linear_model, loader, optimizer, criterion)
    print("done")


    print("done")


if __name__ == '__main__':
    main()