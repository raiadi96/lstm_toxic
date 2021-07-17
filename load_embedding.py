import numpy as np
import os

def load_embedding(path):
    print("Loading word Vectors...")
    word2vec = {}
    with open(os.path.join(path), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    print("Found {0} word Vectors".format(len(word2vec)))
    return word2vec


