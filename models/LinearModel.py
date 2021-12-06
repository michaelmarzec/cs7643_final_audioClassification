import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        This model is just used as a baseline for setting up the training
        and evaluation infrastructure
        :param input_dim: The size of the input
        :param hidden_size: The size of the hidden layer
        :param num_classes: The size of the output
        """
        super(LinearModel, self).__init__()
        
        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

        print(x.get_device())

        x = self.flatten(x)
        out = self.stack(x)
        return out
