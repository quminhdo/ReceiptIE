import torch
import numpy as np
import json
from collections import namedtuple
from gensim.models import KeyedVectors

def json_file_to_pyobj(filename):
    def _json_object_hook(d): return namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    with open(filename) as f:
        data = f.read()
    print(data)
    return json2obj(data)

def get_embeddings(vocab_file, word2vec_file):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = f.read().splitlines()
    model = KeyedVectors.load(word2vec_file)
    embeddings = []
    for word in vocab:
        try:
            vector = model[word]
        except:
            vector = model["[UNK]"]
        embeddings.append(vector)
    return torch.from_numpy(np.stack(embeddings, axis=0))