import json
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
from pprint import pformat
from importlib import import_module
from vocab import Vocab
from preprocess_data import dann

import dataset_woz
import dataset_dstc

# dann = "data/demo/ann"

def load_dataset(args, splits=('train', 'dev', 'test')):
    data_type = args.data_type
    if data_type == 'woz':
        Dataset = dataset_woz.Dataset
        Ontology = dataset_woz.Ontology
        dann = args.dann_woz
    else:
        Dataset = dataset_dstc.Dataset
        Ontology = dataset_dstc.Ontology
        dann = args.dann_dstc

    with open(os.path.join(dann, 'ontology.json')) as f:
        ontology = Ontology.from_dict(json.load(f))
    with open(os.path.join(dann, 'vocab.json')) as f:
        vocab = Vocab.from_dict(json.load(f))
    with open(os.path.join(dann, 'emb.json')) as f:
        E = json.load(f)
    # with open(os.path.join(dann, 'description.json')) as f:
    #     des = json.load(f)
    des = None
    dataset = {}
    for split in splits:
        with open(os.path.join(dann, '{}.json'.format(split))) as f:
            # logging.warn('loading split {}'.format(split))  #loading warning
            dataset[split] = Dataset.from_dict(json.load(f))

    logging.info('dataset sizes: {}'.format(pformat({k: len(v) for k, v in dataset.items()})))
    return dataset, ontology, vocab, E, des


def get_models():
    # tmp = [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']
    return ['glad']
    # return tmp


def load_model(model, *args, **kwargs):
    Model = import_module('{}'.format(model)).Model
    model = Model(*args, **kwargs)
    # logging.info('loaded model {}'.format(Model))  #model loaded path warning
    return model
