import argparse

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import glob
import os
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch.utils import data

from dataset import FunctionIdentificationDataset
from single_elf import FunctionIdentificationELF
from model import CNNModel
from elftools.elf.elffile import ELFFile

kernel_size = 20


def main(root_directory):

    glob_path = glob.glob(os.path.join(root_directory, "*"))

    print("Loading model\n")
    model = torch.load('./model_gcc')

    for binary_path in glob_path:
        print(binary_path.split("/")[-1])
        binary_elf = FunctionIdentificationELF(binary_path, block_size=1000, padding_size=kernel_size - 1)


        print("Ground Truth\n")
        start_list = ground_truth(binary_path)
        #with open(binary_path+'_ground_truth.txt', 'w') as f:
        #    f.write('\n'.join(str(i) for i in start_list))


        print("All Instructions\n")
        all_instructions, all_tags = get_all_instructions(binary_path)
        #with open(binary_path+'_all_instructions.txt', 'w') as f:
        #    f.write('\n'.join(str(all_instructions[i]) + " " + str(all_tags[i]) for i in range(len(all_instructions))))
            #f.write('\n'.join(str(i) for i in (all_tags)))


        print("Testing")
        all_tag_scores = test_model(model, binary_elf)
        #with open(binary_path+'_cnn_prediction.txt', 'w') as f:
        #    f.write('\n'.join(str(i) for i in all_tag_scores))

        with open(binary_path+'_cnn_prediction.txt', 'w') as f:
            f.write('\n'.join(str(all_instructions[i]) + " " + str(all_tags[i]) + " " + str(all_tag_scores[i]) for i in range(len(all_instructions))))

def test_model(model, test_dataset):
    test_loader = data.DataLoader(test_dataset)
    #print("length of test_loader", len(test_loader))
    model.eval()
    with torch.no_grad():
        all_tags = []
        all_tag_scores = []
        for sample, tags in test_loader:
            sample = sample[0]
            tags = tags[0]

            tag_scores = model(sample)

            all_tags.extend(tags.numpy())
            all_tag_scores.extend(tag_scores.numpy())

        all_tags = numpy.array(all_tags)
        all_tag_scores = numpy.array(all_tag_scores).argmax(axis=1)

        print('all_tag_scores', len(all_tag_scores))
        print('all_tags', len(all_tags))

        accuracy = accuracy_score(all_tags, all_tag_scores)
        pr = precision_score(all_tags, all_tag_scores)
        recall = recall_score(all_tags, all_tag_scores)
        f1 = f1_score(all_tags, all_tag_scores)

        print("accuracy: {}".format(accuracy))
        print("pr: {}".format(pr))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))

        return all_tag_scores

def ground_truth(binary_path):
    with open(binary_path, "rb") as binary_file:
        binary_elf = ELFFile(binary_file)

        symbol_table = binary_elf.get_section_by_name(".symtab")

        counter = 0
        start_list = []

        for symbol in symbol_table.iter_symbols():
            if symbol["st_info"]["type"] == "STT_FUNC" and symbol["st_size"] != 0:
                counter += 1
                if hex(symbol["st_value"]) not in start_list:
                    start_list.append(hex(symbol["st_value"]))

        print('counter', counter)
        print('length of start_list', len(start_list))

    return start_list

def get_all_instructions(binary_path):

    binary_elf_class = FunctionIdentificationELF(binary_path, block_size=1000, padding_size=19)

    with open(binary_path, "rb") as binary_file:
        binary_elf = ELFFile(binary_file)

        all_tags = binary_elf_class._generate_tags(binary_elf)
        all_instructions = binary_elf_class._generate_all_instruction(binary_elf)

        print('length of all_tags', len(all_tags))
        print('length of all_instructions', len(all_instructions))

    return all_instructions, all_tags

if __name__ == '__main__':
    #root_directory = "/home/gavin/data_HDD/function_identification/elf/elf-x86-64/binary"
    #root_directory = "/home/gavin/data_HDD/function_identification/lookback-corpus/testsuite-utils-gcc-unstrip"
    #root_directory = "/home/gavin/data_HDD/function_identification/lookback-corpus/testsuite-spec2017-gcc-unstrip"
    root_directory = "/home/gavin/data_HDD/function_identification/lookback-corpus/testsuite-gnutils-gcc-unstrip"
    main(root_directory)