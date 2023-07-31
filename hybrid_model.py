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

# %%
# combine ghidra and CNN resutls
cnn_folder = "/home/gavin/data/function_identification/elf/cnn"
ghidra_folder = "/home/gavin/data/function_identification/elf/ghidra"

cnn_file_list = glob.glob(os.path.join(cnn_folder, "*"))
ghidra_file_list = glob.glob(os.path.join(ghidra_folder, "*"))

# KNN x['cnn','ghidra']
# y ['ground']

x = []
y = []

# %%
counter = 0

for ghidra_file in ghidra_file_list[0:100]:
    ghidra_name = ghidra_file.split('/')[-1].split('.')[0]
    index = -1
    for cnn_file in cnn_file_list:
        index += 1
        if ghidra_name in cnn_file:
            cnn_name = cnn_file.split('/')[-1].split('.')[0]
            #print(ghidra_name)
            #print(index, cnn_name)

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

            print(len(ghidra_list))
            counter += len(ghidra_list)

            cnn_file_data['Ghidra'] = ghidra_list

            for i in cnn_file_data.iloc:
                x.append([i['CNN'], i['Ghidra']])
                y.append(i['Ground'])

print(counter)


# %%
print(len(x))
print(len(y))

# %% [markdown]
# ### KNN

# %%
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x, y)

# %%
knnPickle = open('knn-100', 'wb')  
# source, destination 
pickle.dump(neigh, knnPickle)  
# close the file
knnPickle.close()

# %%


# %%
## load test dataset
# combine ghidra and CNN resutls
cnn_folder2 = "/home/gavin/data/function_identification/lookback-corpus/cnn"
ghidra_folder2 = "/home/gavin/data/function_identification/lookback-corpus/ghidra"

cnn_file_list2 = glob.glob(os.path.join(cnn_folder2, "*"))
ghidra_file_list2 = glob.glob(os.path.join(ghidra_folder2, "*"))

# KNN x['cnn','ghidra']
# y ['ground']

x2 = []
y2 = []

# %%
counter = 0

for ghidra_file in ghidra_file_list2[0:5]:
    ghidra_name = ghidra_file.split('/')[-1].split('.')[0]
    index = -1
    for cnn_file in cnn_file_list2:
        index += 1
        if ghidra_name in cnn_file:
            cnn_name = cnn_file.split('/')[-1].split('.')[0]
            print(ghidra_name)
            print(index, cnn_name)

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

            print(len(ghidra_list))
            counter += len(ghidra_list)

            cnn_file_data['Ghidra'] = ghidra_list

            for i in cnn_file_data.iloc:
                x2.append([i['CNN'], i['Ghidra']])
                y2.append(i['Ground'])

print(counter)

# %%
y_pred2 = neigh.predict(x2)
print(confusion_matrix(y2,y_pred2))
print(classification_report(y2,y_pred2))