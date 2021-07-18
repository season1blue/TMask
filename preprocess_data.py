#!/usr/bin/env python
import os
import json
import logging
import requests
from tqdm import tqdm
from vocab import Vocab
from embeddings import GloveEmbedding, KazumaCharEmbedding

import dataset_woz
import dataset_dstc

data_type = "dstc"

if data_type == "woz":
    Dataset = dataset_woz.Dataset
    Ontology = dataset_woz.Ontology
    root_dir = os.path.dirname(__file__)
    data_dir = os.path.join(root_dir, 'data', 'woz')  # annonate the demo dataset
else:
    Dataset = dataset_dstc.Dataset
    Ontology = dataset_dstc.Ontology
    root_dir = os.path.dirname(__file__)
    data_dir = os.path.join(root_dir, 'data', 'DSTC',"dstc")  # annonate the demo dataset


draw = os.path.join(data_dir, 'raw')
dann = os.path.join(data_dir, 'ann')
splits = ['dev', 'train', 'test']
# splits2 = ['test']

# def download(url, to_file):
#     r = requests.get(url, stream=True)
#     with open(to_file, 'wb') as f:
#         for chunk in r.iter_content(chunk_size=1024):
#             if chunk:
#                 f.write(chunk)
#
#
# def missing_files(d, files):
#     # return not all([os.path.isfile(os.path.join(d, '{}.json'.format(s))) for s in files])
#     return True


if __name__ == '__main__':
    # if missing_files(draw, splits):
    #     print("still request a raw dataset")
        # if not os.path.isdir(draw):
        #     os.makedirs(draw)
        # download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_train_en.json', os.path.join(draw, 'train.json'))
        # download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_validate_en.json', os.path.join(draw, 'dev.json'))
        # download('https://github.com/nmrksic/neural-belief-tracker/raw/master/data/woz/woz_test_en.json', os.path.join(draw, 'test.json'))

    if not os.path.isdir(dann):
        os.makedirs(dann)
    dataset = {}
    ontology = Ontology()
    vocab = Vocab()
    vocab.word2index(['<sos>', '<eos>'], train=True)
    for s in splits:
        fname = '{}.json'.format(s)
        logging.warning('Annotating {}'.format(s))
        dataset[s] = Dataset.annotate_raw(os.path.join(draw, fname))  ##error
        # print(dataset[s].to_dict())  还没有num
        dataset[s].numericalize_(vocab)
        ontology = ontology + dataset[s].extract_ontology()
        with open(os.path.join(dann, fname), 'wt') as f:
            json.dump(dataset[s].to_dict(), f)


    ontology.numericalize_(vocab)

    with open(os.path.join(dann, 'ontology.json'), 'wt') as f:
        json.dump(ontology.to_dict(), f)
    with open(os.path.join(dann, 'vocab.json'), 'wt') as f:
        json.dump(vocab.to_dict(), f)
    # with open(os.path.join(dann, 'description.json'), 'r') as f:
    #     des = json.load(f)


    logging.warning('Computing word embeddings')
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for w in tqdm(vocab._index2word):
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(os.path.join(dann, 'emb.json'), 'wt') as f:
        json.dump(E, f)
