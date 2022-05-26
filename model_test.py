import os
from xml.etree.ElementInclude import include
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import sys
from seq2tensor import s2t
from tqdm import tqdm

import pandas as pd

# Network parameters
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=None, help='Filepath to a pretrained model.')
parser.add_argument('--dic', type=str, default='yeast', help='Name of the file containing data dictionary.')
parser.add_argument('--act', type=str, default='yeast', help='Name of the file containing data actions.')
parser.add_argument('--emb', type=int, default=3, help='Embedding type: 0=oneshot, 1=string_vec5,'
                                                       '2=CTCoding_onehot, 3=vec5_CTC.')
parser.add_argument('--csv', type=str, default='results/out.csv', help='Name of the output csv file.')

args = parser.parse_args()
id2seq_file = args.dic

def remove_space(line):
    for i,l in enumerate(line):
        if ' ' in l:
            line[i] = line[i][:str.find(l, ' ')]
    return line

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    seqs.append(line[1])
    index += 1
seq_array = []
id2_aid = {}
sid = 0

seq_size = 2000
emb_files = [os.path.expanduser('embeddings/default_onehot.txt'), os.path.expanduser('embeddings/string_vec5.txt'),
             os.path.expanduser('embeddings/CTCoding_onehot.txt'), os.path.expanduser('embeddings/vec5_CTC.txt')]
hidden_dim = 25
n_epochs = 50

# ds_file = os.path.expanduser('yeast/preprocessed/protein.actions.tsv')
# ds_file = os.path.expanduser('../Models_and_Datasets/ath/protein.actions.tsv')
ds_file = os.path.expanduser(args.act)
label_index = 2
sid1_index = 0
sid2_index = 1
seq2t = s2t(emb_files[int(args.emb)])

max_data = -1
limit_data = max_data > 0
raw_data = []
skip_head = True
x = None
count = 0
ids = []

for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
    line = remove_space(line)
    if len(line) != 3:
        print("bug with the line:",line)
        continue
    
    if id2index.get(line[sid1_index]) is None or id2index.get(line[sid2_index]) is None:
        continue
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]
    raw_data.append(line)
    ids += [line.copy()] # for the csv file
    if limit_data:
        count += 1
        if count >= max_data:
            break
# print("raw",raw_data[0])

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

class_map = {'0': 1, '1': 0}
class_labels = np.zeros((len(raw_data), 2))
for i in range(len(raw_data)):
    class_labels[i][class_map[raw_data[i][label_index]]] = 1.
model = keras.models.load_model(args.model)
# model.summary()
labels = []
pred = []

labels_roc = []
pred_roc = []

pred_csv = []


prev = 0
for i in tqdm(range(128, len(seq_index1)+127, 128)):
    s1 = seq_tensor[seq_index1[prev:i]]
    s2 = seq_tensor[seq_index2[prev:i]]
    p = model.predict([s1, s2])
    l = class_labels[prev:i]
    if prev == 0:
        pred_roc = p.copy().flatten()
        labels_roc = l.copy().flatten()
        pred = np.argmax(p,axis=-1).flatten()
        labels = np.argmax(l,axis=-1).flatten()
        for j in range(prev,i): # for the csv save file
            ids[j] = ids[j] + [p[:,0][j-prev]]
    else:
        pred_roc = np.concatenate((pred_roc, p.copy().flatten()))
        labels_roc = np.concatenate((labels_roc, l.copy().flatten()))
        pred = np.concatenate((pred, np.argmax(p,axis=-1).flatten()))
        labels = np.concatenate((labels, np.argmax(l,axis=-1).flatten()))
        pred_csv = np.concatenate((pred_csv, p[:,0]))
        for j in range(prev,min(len(ids),i)): # for the csv save file
            if len(p[:,0]) == (j-prev):
                print(len(ids))
                print(len(seq_index1))
                print(len(p[:,0]))
            ids[j] = ids[j] + [p[:,0][j-prev]]
    prev = i

# create the csv file
dic = {"p1":[],"p2":[],"truth":[],"prediction":[]}
for i in range(len(ids)):
    dic["p1"] = dic["p1"] + [ids[i][0]]
    dic["p2"] = dic["p2"] + [ids[i][1]]
    dic["truth"] = dic["truth"] + [int(ids[i][2])]
    dic["prediction"] = dic["prediction"] + [ids[i][3]]
df = pd.DataFrame(dic)
df.to_csv(args.csv, index=False)

auc = roc_auc_score(labels_roc, pred_roc)
print("Test ROC AUC score: ", auc)

acc = accuracy_score(labels, pred)
print("Test accuracy score:", acc)

pre = precision_score(labels, pred)
print("Test precision score:", pre)

rec = recall_score(labels, pred)
print("Test recall score:", rec)
