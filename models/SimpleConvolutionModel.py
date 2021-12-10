import torch
import torch.nn as nn


class SimpleConvolutionModel(nn.Module):
    def __init__(self, starting_kernel_size=3, dropout_rate=0.2):
        """
        This model is just used as a baseline for setting up the training
        and evaluation infrastructure
        :param input_dim: The size of the input
        :param hidden_size: The size of the hidden layer
        :param num_classes: The size of the output
        """
        super(SimpleConvolutionModel, self).__init__()

        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.flatten = nn.Flatten()
        self.layer1 = nn.Conv1d(1, 32, starting_kernel_size)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv1d(32, 64, starting_kernel_size + 2)
        self.layer4 = nn.Dropout(dropout_rate)
        self.layer5 = nn.MaxPool1d(starting_kernel_size + 2)
        self.layer6 = nn.Conv1d(64, 128, starting_kernel_size + 4)
        self.layer7 = nn.Dropout(dropout_rate)
        self.layer8 = nn.Linear(31744, 16384)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Linear(16384, 2)

    def forward(self, x):
        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

        x = self.flatten(x)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = torch.flatten(out, 1)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        return out
