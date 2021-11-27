from utils import utils
from utils import dataloader
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)

    training_data = utils.load_pytorch_tensor('./utils/balanced_train_data.pt')
    training_label = utils.load_pytorch_tensor('./utils/balanced_train_label.pt')

    dataset = dataloader.MusicDataset(training_data, training_label)


if __name__ == '__main__':
    main()