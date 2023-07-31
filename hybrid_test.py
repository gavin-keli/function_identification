# %% [markdown]
# ## Hybrid Model

# %%
import os
import glob
import pickle 
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix



neigh = pickle.load(open('knn', 'rb'))
#neigh = pickle.load(open('knn-100', 'rb'))


# %%
## load test dataset
# combine ghidra and CNN resutls
cnn_folder2 = "/home/gavin/data/function_identification/lookback-corpus/cnn"

#ghidra_folder2 = "/home/gavin/data/function_identification/lookback-corpus/ghidra/O1"
#ghidra_folder2 = "/home/gavin/data/function_identification/lookback-corpus/ghidra/O2"
ghidra_folder2 = "/home/gavin/data/function_identification/lookback-corpus/ghidra/O3"

cnn_file_list2 = glob.glob(os.path.join(cnn_folder2, "*"))
ghidra_file_list2 = glob.glob(os.path.join(ghidra_folder2, "*"))

# KNN x['cnn','ghidra']
# y ['ground']

x2 = []
y2 = []

# %%
counter = 0
counter_length = 0

for ghidra_file in ghidra_file_list2[0:25]:
    ghidra_name_long = ghidra_file.split('/')[-1].split('.')[0]
    ghidra_name = ghidra_name_long[0:-6]
    index = -1
    for cnn_file in cnn_file_list2:
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

            for i in cnn_file_data.iloc:
                x2.append([i['CNN'], i['Ghidra']])
                y2.append(i['Ground'])

print(counter)
print(counter_length)

# %%
y_pred2 = neigh.predict(x2)
print(confusion_matrix(y2,y_pred2))
print(classification_report(y2,y_pred2))