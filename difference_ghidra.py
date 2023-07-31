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

cnn_ghidra_file_list = glob.glob(os.path.join(cnn_ghidra, "*"))

cnn = []
ghidra = []

counter = 0

O1_differences = 0
O1_number_of_lines = 0

O2_differences = 0
O2_number_of_lines = 0

O3_differences = 0
O3_number_of_lines = 0

# O1 vs O2 vs O3

for cnn_ghidra_file in cnn_ghidra_file_list:
         
    cnn_ghidra_file_data = pd.read_csv(cnn_ghidra_file, header=None, names=["Address", "Ground", "CNN", "Ghidra"])

    if '-O1_' in cnn_ghidra_file:
        O1_number_of_lines += len(cnn_ghidra_file_data)
    elif '-O2_' in cnn_ghidra_file:
        O2_number_of_lines += len(cnn_ghidra_file_data)
    elif '-O3_' in cnn_ghidra_file:
        O3_number_of_lines += len(cnn_ghidra_file_data)
    else:
        print("ERROR FOUND!")

    for row in cnn_ghidra_file_data.iloc:
        #print(row)
        CNN_result = row[2]
        Ghidra_result = row[3]

        if '-O1_' in cnn_ghidra_file:
            if CNN_result != Ghidra_result:
                #print('CNN_result', CNN_result, 'Ghidra_result', Ghidra_result)
                O1_differences += 1
        
        elif '-O2_' in cnn_ghidra_file:
            if CNN_result != Ghidra_result:
                #print('CNN_result', CNN_result, 'Ghidra_result', Ghidra_result)
                O2_differences += 1

        elif '-O3_' in cnn_ghidra_file:
            if CNN_result != Ghidra_result:
                #print('CNN_result', CNN_result, 'Ghidra_result', Ghidra_result)
                O3_differences += 1
                     
    counter += 1
    print(counter)

print(counter)


print('O1_differences', O1_differences)
print('O1_number_of_lines', O1_number_of_lines)

print('O2_differences', O2_differences)
print('O2_number_of_lines', O2_number_of_lines)

print('O3_differences', O3_differences)
print('O3_number_of_lines', O3_number_of_lines)