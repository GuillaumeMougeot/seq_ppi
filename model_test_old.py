import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score
import sys
from seq2tensor import s2t
from tqdm import tqdm

# Network parameters
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default=None, help='Filepath to a pretrained model.')
parser.add_argument('--data', type=str, default='yeast', help='Name of the folder containing data.')
parser.add_argument('--emb', type=int, default=3, help='Embedding type: 0=oneshot, 1=string_vec5,'
                                                       '2=CTCoding_onehot, 3=vec5_CTC.')

args = parser.parse_args()
id2seq_file = args.data

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
emb_files = [os.path.expanduser('~/PIPR/seq_ppi/embeddings/default_onehot.txt'), os.path.expanduser('~/PIPR/seq_ppi/embeddings/string_vec5.txt'),
             os.path.expanduser('~/PIPR/seq_ppi/embeddings/CTCoding_onehot.txt'), os.path.expanduser('~/PIPR/seq_ppi/embeddings/vec5_CTC.txt')]
hidden_dim = 25
n_epochs = 50

ds_file = os.path.expanduser('~/PIPR/seq_ppi/yeast/preprocessed/protein.actions.tsv')
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

for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').split('\t')
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
    if limit_data:
        count += 1
        if count >= max_data:
            break
print(len(raw_data))

dim = seq2t.dim
seq_tensor = np.array([seq2t.embed_normalized(line, seq_size) for line in tqdm(seq_array)])

seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])

print(seq_index1[:10])

class_map = {'0': 1, '1': 0}
print(class_map)
class_labels = np.zeros((len(raw_data), 2))
for i in range(len(raw_data)):
    class_labels[i][class_map[raw_data[i][label_index]]] = 1.
model = keras.models.load_model(args.model)

labels = []
pred = []

prev = 0
for i in range(128, len(seq_index1)+127, 128):
    s1 = seq_tensor[seq_index1[prev:i]]
    s2 = seq_tensor[seq_index2[prev:i]]
    if prev == 0:
        pred = model.predict([s1, s2]).flatten()
        labels = l[prev:i]
    else:
        pred = np.concatenate((pred, model.predict([s1, s2]).flatten()), dtype=np.float16)
        labels = np.concatenate((labels, l[prev:i]))
    prev = i

auc = roc_auc_score(labels, pred)
print("Test ROC AUC score: ", auc)
