import os.path

import torch
import tfrecord
import numpy as np
from os import walk

# Tqdm progress bar
from tqdm import tqdm_notebook

def read_tf_records(filename : str) -> torch.Tensor:
    '''
    The output of this function should be x_data, y_data
    :param filename: filename of the tfrecord
    :return: x_data: Nx10x128 representing 10 seconds of feature data of the audio file
             y_data: Nx4 class label between 0 and 526 inclusive representing the
                     sound (164 is the drum)
    '''

    context_description = {"video_id": "byte", "start_time_seconds": "float", "end_time_seconds": "float", "labels": "int"}
    sequence_description = {"audio_embedding": "byte"}

    loader = tfrecord.tfrecord_loader(filename, None, context_description, sequence_description=sequence_description)

    x_data = None
    y_data = None
    skip_count = 0
    for context, sequence in loader:
        if x_data is not None:
            list_data = sequence['audio_embedding']
            if len(list_data) != 10:
                skip_count += 1
                continue
            list_data = np.reshape(list_data, (1, 10, 128))
            x_data = np.vstack((x_data, list_data))
        else:
            list_data = sequence['audio_embedding']
            if len(list_data) != 10:
                skip_count += 1
                continue
            x_data = np.reshape(list_data, (1, 10, 128))

        if y_data is not None:
            label_data = context['labels']
            label_data = np.resize(label_data, (1, 4))
            y_data = np.vstack((y_data, label_data))
        else:
            label_data = context['labels']
            label_data = np.resize(label_data, (1, 4))
            y_data = label_data

    return x_data, y_data


def load_pytorch_tensor(filename: str):
    """
    Loads a saved tensor
    :param filename: Filename of the tensor to load
    :return: Loaded tensor
    """
    tensor = torch.load(filename)
    return tensor


def train(model, dataloader, optimizer, criterion):
    """
    Stolen from CS7643 Assignment 4 for training the model
    :param model: Torch Model
    :param dataloader: For the progress bar of the notebook
    :param optimizer: Optimizer for gradient descent
    :param criterion: Loss calculation (should be cross entropy loss)
    :return: loss
    """
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        source = data.src.transpose(1, 0)
        target = data.trg.transpose(1, 0)

        translation = model(source)
        translation = translation.reshape(-1, translation.shape[-1])
        target = target.reshape(-1)

        optimizer.zero_grad()
        loss = criterion(translation, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.data
        progress_bar.set_description_str(
            "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    return total_loss, total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    """
    Stolen from CS7643 Assignment 4 for evaluating a model's performance
    :param model: Pytorch Model
    :param dataloader: For progress bar
    :param criterion: Cross entropy loss for us
    :return: Loss
    """
    # Set the model to eval mode to avoid weights update
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            source = data.src.transpose(1, 0)
            target = data.trg.transpose(1, 0)

            translation = model(source)
            translation = translation.reshape(-1, translation.shape[-1])
            target = target.reshape(-1)

            loss = criterion(translation, target)
            total_loss += loss
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


if __name__ == '__main__':
    balanced_train_set = []
    for (dirpath, dirnames, filenames) in walk('./../features/audioset_v1_embeddings/bal_train'):
        for files in filenames:
            realpath = os.path.join(dirpath, files)
            balanced_train_set.append(realpath)
        break

    x_data = None
    y_data = None
    for record in balanced_train_set:
        x_data_in, y_data_in = read_tf_records(record)

        if x_data is not None:
            if x_data_in is not None:
                x_data = np.vstack((x_data, x_data_in))
                y_data = np.vstack((y_data, y_data_in))
        else:
            x_data = x_data_in
            y_data = y_data_in

    torch.save(x_data, 'balanced_train_data.pt')
    torch.save(y_data, 'balanced_train_label.pt')