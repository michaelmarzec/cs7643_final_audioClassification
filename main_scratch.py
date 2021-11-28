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

    # make this multi-classification problem a binary classification problem
    training_label, count = utils.convert_multiclass_to_binary(0, training_label)
    print("Total of " + str(count) + " positive examples out of " + str(training_label.shape[0]) + " samples")
    # training_label = np.reshape(training_label, (training_label.shape[0], 1))

    training_data = np.float32(training_data)
    # training_label = np.float32(training_label)

    data_train, label_train, data_val, label_val = utils.split_data_train_val(training_data, training_label)

    train_dataset = dataloader.MusicDataset(data_train, label_train)
    val_dataset = dataloader.MusicDataset(data_val, label_val)

    # The Dataloader class handles all the shuffles for you
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # for the linear model, the input will be all 10seconds of audio
    # stacked into one layer - therefore input dimension
    # is 10 * 128
    linear_model = LinearModel.LinearModel(10*128, 64, 2)
    optimizer = optim.Adam(linear_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    for epoch_idx in range(10):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx + 1))
        print("-----------------------------------")

        train_loss, avg_train_loss = utils.train(linear_model, train_loader, optimizer, criterion)
        scheduler.step(train_loss)

        val_loss, avg_val_loss = utils.evaluate(linear_model, val_loader, criterion)

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        # avg_val_loss = 0.0
        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("done")


    print("done")


if __name__ == '__main__':
    main()