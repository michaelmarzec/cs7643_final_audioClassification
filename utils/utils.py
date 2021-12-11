import os.path

import torch
import tfrecord
import numpy as np
from os import walk

import torchmetrics
from torchmetrics import ConfusionMatrix
from torchmetrics import F1

# Tqdm progress bar
from tqdm import tqdm_notebook

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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


def split_data_train_val(dataset, labels, percent_val=0.2):
    """
    Take a dataset (data + labels) and return two sets, one for t
    training and one for validation
    :param dataset: Data portion only
    :param labels: Label portio only
    :param percent_val: Split, this much data becomes validation
    :return: 4 tensors, data_train, label_train, data_val, label_val
    """
    total_length = dataset.shape[0]
    shuffled_indices = np.array(np.arange(total_length))
    np.random.shuffle(shuffled_indices)

    train_len = int(total_length * (1.0 - percent_val))
    shuffled_data = dataset[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    data_train = shuffled_data[:train_len]
    label_train = shuffled_labels[:train_len]
    data_val = shuffled_data[train_len:]
    label_val = shuffled_labels[train_len:]
    return data_train, label_train, data_val, label_val


def convert_multiclass_to_binary(wanted_class: int, array: np.ndarray):
    print(array.shape)
    return_array = np.zeros((array.shape[0], 2), dtype=float)
    count = 0
    for index in range(0, array.shape[0]):
        if wanted_class in array[index, :]:
            return_array[index, 1] = 1
            count += 1
        else:
            return_array[index, 0] = 1

    return return_array, count


def train(model, dataloader, optimizer, criterion):
    """
    Stolen from CS7643 Assignment 4 for training the model
    :param model: Torch Model
    :param dataloader: For the progress bar of the notebook
    :param optimizer: Optimizer for gradient descent
    :param criterion: Loss calculation (should be cross entropy loss)
    :return: loss
    """
    dataloader = iter(dataloader)
    model.train()

    # Record total loss
    total_loss = 0.

    # Get the progress bar for later modification
    progress_bar = tqdm_notebook(dataloader, ascii=True)

    # Mini-batch training
    for batch_idx, data in enumerate(progress_bar):
        input_data = data[0].to(device)
        correct_labels = data[1].to(device)

        prediction = model(torch.tensor(input_data, dtype=torch.float32).to(device))

        optimizer.zero_grad()
        loss = criterion(prediction, correct_labels)
        loss.backward()
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
    dataloader = iter(dataloader)
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            input_data = data[0].to(device)
            correct_labels = data[1].to(device)

            prediction = model(input_data)

            loss = criterion(prediction, correct_labels)
            total_loss += loss
            progress_bar.set_description_str(
                "Batch: %d, Loss: %.4f" % ((batch_idx + 1), loss.item()))

    avg_loss = total_loss / len(dataloader)
    return total_loss, avg_loss


def evaluate_with_metrics(model, dataloader):
    """
    Adapted from CS7643 to evaluate and return some metrics on test data
    :param model: Model to be evaluated
    :param dataloader: Data to be loaded
    :return: Bunch of metrics
    """
    # Set the model to eval mode to avoid weights update
    dataloader = iter(dataloader)
    model.eval()

    # declare the metric keepers
    conf_mat = ConfusionMatrix(num_classes=2)
    f1_score = F1(num_classes=2)
    total_accuracy = 0.0

    with torch.no_grad():
        # Get the progress bar
        progress_bar = tqdm_notebook(dataloader, ascii=True)
        for batch_idx, data in enumerate(progress_bar):
            input_data = data[0].to('cpu')
            correct_labels = data[1].to('cpu')

            prediction = model(input_data)

            conf_mat.update(prediction, correct_labels)
            f1_score.update(prediction, correct_labels)
            accuracy = torchmetrics.functional.accuracy(prediction, correct_labels)
            total_accuracy += accuracy
            progress_bar.set_description_str(
                "Batch: %d" % (batch_idx + 1))

    avg_accuracy = total_accuracy / len(dataloader)
    final_conf_mat = conf_mat.compute()
    final_f1_score = f1_score.compute()
    return avg_accuracy, final_conf_mat, final_f1_score


def create_training_tensors():
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

def create_eval_tensors():
    eval_set = []
    for (dirpath, dirnames, filenames) in walk('./../features/audioset_v1_embeddings/eval'):
        for files in filenames:
            realpath = os.path.join(dirpath, files)
            eval_set.append(realpath)
        break

    x_data = None
    y_data = None
    for record in eval_set:
        x_data_in, y_data_in = read_tf_records(record)

        if x_data is not None:
            if x_data_in is not None:
                x_data = np.vstack((x_data, x_data_in))
                y_data = np.vstack((y_data, y_data_in))
        else:
            x_data = x_data_in
            y_data = y_data_in

    torch.save(x_data, 'eval_data.pt')
    torch.save(y_data, 'eval_label.pt')

if __name__ == '__main__':
    create_eval_tensors()