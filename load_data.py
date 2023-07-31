import argparse

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch.utils import data

from dataset import FunctionIdentificationDataset
from model import CNNModel

kernel_size = 20


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("dataset_path", help="Path to the directory with the binaries for the dataset "
                                                      "(e.g ~/elf/elf_32")
    args = argument_parser.parse_args()

    print("Preprocessing")
    dataset = FunctionIdentificationDataset(args.dataset_path, block_size=1000, padding_size=kernel_size - 1)

    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])


    train_loader = data.DataLoader(train_dataset, shuffle=True)

    for sample, tags in tqdm.tqdm(train_loader):
        #sample = sample[0]
        #tags = tags[0]

        print('Sample', sample)
        print('Tags', tags)

        break


if __name__ == '__main__':
    main()
