import torch
import torch.nn as nn


class SimpleRecurentModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dimensions, n_layers):
        """
        This model is just used as a baseline for setting up the training
        and evaluation infrastructure
        :param input_dim: The size of the input
        :param hidden_size: The size of the hidden layer
        :param num_classes: The size of the output
        """
        super(SimpleRecurentModel, self).__init__()

        # params
        self.hidden_dimensions = hidden_dimensions
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_dimensions, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dimensions, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)

        #reshape for fully connected (#####UPDATE)
        out = out.contiguous().view(-1, self.hidden_dimensions)
        out = self.fc(out)

        return out, hidden

