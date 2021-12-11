import torch
import torch.nn as nn


class SimpleRecurrentModel(nn.Module):
    def __init__(self, stack_size=3, hidden_size=64, dropout=0.1):
        """
        Sets of a recurrent LSTM model
        :param stack_size: NUmber of layers of LSTMs stacked together
        """
        super(SimpleRecurrentModel, self).__init__()

        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
        self.flatten = nn.Flatten()

        self.recurrent = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=stack_size, dropout=dropout, batch_first=True)
        self.linear_one = nn.Linear(in_features=hidden_size, out_features=512)
        self.relu = nn.ReLU()
        self.linear_two = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        # code inspired by
        # https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

        # since this is time series - we push forward each time slice of data
        # individually through the recurrent network - these shape of incoming
        # data is batch_size X time_slice X data - we want to push batch_size X data
        # through the network in a loop
        # for time_slice in range(x.shape[1]):
        # time_slice = x[:, time_slice, :]
        output, hidden = self.recurrent(x)

        # we take the last real prediction the network made, because the
        # last step was just <eos>
        output = output[:, -2, :]

        # now take the last output of the LSTM and feed it through the linear network
        out = self.linear_one(output)
        out = self.relu(out)
        out = self.linear_two(out)

        return out
