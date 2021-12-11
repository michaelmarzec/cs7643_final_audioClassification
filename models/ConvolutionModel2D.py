import torch
import torch.nn as nn


class ConvolutionModel(nn.Module):
    def __init__(self, starting_kernel_size=2, dropout_rate=0.2):
        """
        This model is just used as a baseline for setting up the training
        and evaluation infrastructure
        :param input_dim: The size of the input
        :param hidden_size: The size of the hidden layer
        :param num_classes: The size of the output
        """
        super(ConvolutionModel, self).__init__()

        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        num_in_channels = 1
        num_out_channels=3
        flat_size = 558 # kernel size 2
        # flat_size = 180 # kernel size 3
        # flat_size = 600
        self.layer1 = nn.Conv2d(in_channels=num_in_channels,
                                out_channels=num_out_channels,
                                kernel_size=starting_kernel_size
                                )
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Conv2d(in_channels=num_out_channels,
                                out_channels=num_out_channels,
                                kernel_size=starting_kernel_size
                                )
        self.layer5 = nn.MaxPool2d(2)
        self.layer6 = nn.Conv2d(in_channels=num_out_channels,
                                out_channels=num_out_channels,
                                kernel_size=starting_kernel_size
                                )
        self.flatten_layer = nn.Flatten()
        self.b1 = nn.BatchNorm1d(flat_size)
        self.layer7 = nn.Linear(flat_size, flat_size * 2)
        self.layer8 = nn.Dropout(dropout_rate)
        self.layer9 = nn.Linear(flat_size * 2, 2)

    def forward(self, x):
        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html


        x_reshape = x[:, None, :, :]
        out = self.layer1(x_reshape)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.flatten_layer(out)
        out = self.b1(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        return out
