import os
import glob
import pickle 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score

# combine ghidra and CNN resutls
cnn_ghidra = "/home/gavin/data_HDD/function_identification/lookback-corpus/cnn_ghidra/"
cnn_folder = "/home/gavin/data_HDD/function_identification/lookback-corpus/cnn"

#ghidra_folder = "/home/gavin/data_HDD/function_identification/lookback-corpus/ghidra/O1"
#ghidra_folder = "/home/gavin/data_HDD/function_identification/lookback-corpus/ghidra/O2"
ghidra_folder = "/home/gavin/data_HDD/function_identification/lookback-corpus/ghidra/O3"

cnn_file_list = glob.glob(os.path.join(cnn_folder, "*"))
ghidra_file_list = glob.glob(os.path.join(ghidra_folder, "*"))

# KNN x['cnn','ghidra']
# y ['ground']

x = []
y = []

counter = 0
counter_length = 0

for ghidra_file in ghidra_file_list:
    ghidra_name_long = ghidra_file.split('/')[-1].split('.')[0]
    ghidra_name = ghidra_name_long[0:-6]
    index = -1
    for cnn_file in cnn_file_list:
        index += 1
        cnn_name = cnn_file.split('/')[-1].split('.')[0]

        if ghidra_name == cnn_name[0:-15]:
            print(ghidra_name_long)
            print(cnn_name)

            cnn_file_data = pd.read_csv(cnn_file, header=None, sep=" ", names=["Address", "Ground", "CNN"])
            ghidra_file_data = pd.read_csv(ghidra_file, header=None, sep=" ", names=["Address", "Name"])

            ghidra_list = []
            for row in cnn_file_data['Address']:
                state = 0
                for item in ghidra_file_data['Address']:
                    if int(item, 16) == int(row):
                        ghidra_list.append(1)
                        state = 1
                if state == 0:
                    ghidra_list.append(0)

            counter += 1
            counter_length += len(ghidra_list)
            print(counter)
            print(counter_length)

            cnn_file_data['Ghidra'] = ghidra_list
            cnn_file_data.to_csv(cnn_ghidra + cnn_file.split('/')[-1].split('.txt')[0] + '_ghidra.txt', index=False, header=False)
            

            for i in cnn_file_data.iloc:
                x.append(i['Ghidra'])
                y.append(i['Ground'])

print(counter)
print(counter_length)


accuracy = accuracy_score(y, x)
pr = precision_score(y, x)
recall = recall_score(y, x)
f1 = f1_score(y, x)

print("accuracy: {}".format(accuracy))
print("pr: {}".format(pr))
print("recall: {}".format(recall))
print("f1: {}".format(f1))