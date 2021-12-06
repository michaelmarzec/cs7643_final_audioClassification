import torch
from torch.utils.data import Dataset, DataLoader

# inspired from https://towardsdatascience.com/generating-batch-data-for-pytorch-7435b1a02e21
class MusicDataset(Dataset):
  """
  Because all of our data is globbed together into one massive batch, we use this
  class extension for torch's Dataset to create batches, training, and validation data
  """
  def __init__(self, X, Y, device='cpu'):
    self.X = torch.from_numpy(X).to(device)
    self.Y = torch.from_numpy(Y).to(device)
    if len(X) != len(Y):
      raise Exception("The length of X does not match the length of Y")

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
    _x = self.X[index, :, :]
    _y = self.Y[index]

    return _x, _y