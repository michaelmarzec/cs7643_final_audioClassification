import numpy as np

from torch.utils.data import DataLoader

from models import LinearModel
from models import SimpleConvolutionModel
from models import SimpleRecurrentModel
from utils import utils
from utils import dataloader
import torch
from torch import nn
from torch import optim

import torchmetrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)

    training_data = utils.load_pytorch_tensor('./utils/balanced_train_data.pt')
    training_label = utils.load_pytorch_tensor('./utils/balanced_train_label.pt')

    # make this multi-classification problem a binary classification problem
    training_label, count = utils.convert_multiclass_to_binary(0, training_label)
    print("Total of " + str(count) + " positive examples out of " + str(training_label.shape[0]) + " samples")
    # training_label = np.reshape(training_label, (training_label.shape[0], 1))

    # training_data = utils.add_sos_eos_tokens_data(training_data)
    training_data, training_label = utils.augment_training_data(training_data, training_label)
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
    # model = LinearModel.LinearModel()
    model = SimpleConvolutionModel.SimpleConvolutionModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    for epoch_idx in range(1):
        print("-----------------------------------")
        print("Epoch %d" % (epoch_idx + 1))
        print("-----------------------------------")

        train_loss, avg_train_loss = utils.train(model, train_loader, optimizer, criterion)
        scheduler.step(train_loss)

        val_loss, avg_val_loss = utils.evaluate(model, val_loader, criterion)

        avg_train_loss = avg_train_loss.item()
        avg_val_loss = avg_val_loss.item()
        # avg_val_loss = 0.0
        print("Training Loss: %.4f. Validation Loss: %.4f. " % (avg_train_loss, avg_val_loss))
    print("done")


    # evaluate the model
    model = model

    eval_batch_size = 512

    eval_data = utils.load_pytorch_tensor('./utils/eval_data.pt')
    eval_label = utils.load_pytorch_tensor('./utils/eval_label.pt')

    # make this multi-classification problem a binary classification problem
    eval_label, count = utils.convert_multiclass_to_binary(0, eval_label)
    print("Total of " + str(count) + " positive examples out of " + str(eval_label.shape[0]) + " samples")

    eval_data = np.float32(eval_data)

    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=True)

    # push the eval data through the model
    predictions = model(eval_data)
    acc = torchmetrics.functional.accuracy(predictions, eval_label)

    print("Model achieved an accuracy of %d on evaluation data", str(acc))



if __name__ == '__main__':
    main()