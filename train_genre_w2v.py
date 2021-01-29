import pandas as pd
import json
from gensim.models import Word2Vec
import numpy as np
#from googletrans import Translator


SIZE_W2V = 128


tags_gnr_data = pd.read_csv('tags/w2v_all_tags_gnr.csv', sep='delimiter', header=None, error_bad_lines=False)
num_lines = len(tags_gnr_data)
tags_gnr = [
    list(set([
        str(int(t))
        for t in tags_gnr_data.iloc[idx].tolist()[0].split(',')
        if t
    ]))
    for idx in range(num_lines)
]
num_max_words = max([len(i) for i in tags_gnr])
model = Word2Vec(tags_gnr, size=SIZE_W2V, window=num_max_words, min_count=1, workers=4)

tags_gnr_map = json.load(open('tags/w2v_all_gnr_map.json', 'rb'))
id2word = {str(v): k for k, v in tags_gnr_map.items()}


embedding_matrix = np.zeros((len(model.wv.vocab)+1, SIZE_W2V))
for i, v in id2word.items():
    try:
        embedding_vector = model.wv[i]
        embedding_matrix[int(i)+1] = embedding_vector
    except:
        pass

np.save(f'embedding_matrix_{SIZE_W2V}.npy', embedding_matrix)
