#!/usr/bin/env python3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from utils import load_dataset, get_models, load_model
import os
import logging
import numpy as np
from pprint import pprint
import torch
from random import seed
import time


def run(args):
    # pprint(args)
    logging.basicConfig(level=logging.INFO)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    seed(args.seed)
    dataset, ontology, vocab, Eword, des = load_dataset(args)  # load dataset

    # print(dataset['train'].to_dict())
    model = load_model(args.model, args, ontology, vocab, des)
    model.save_config()
    model.load_emb(Eword)

    model = model.to(model.device)

    if not args.test:
        # logging.info('Starting train')
        model.run_train(dataset['train'], dataset['dev'], args)  #first train


    " originaly, there is evaluation after all the epoch, but it it used by finding the best model in defult files and load the best model. After delete these, we can see the best in train result"
    # if args.resume:
    #     model.load_best_save(directory=args.resume)
    # else:
    #     model.load_best_save(directory=args.dout)  #come here
    # model = model.to(model.device)
    # logging.info('Running dev evaluation')
    # dev_out = model.run_eval(dataset['dev'], args)  # second dev
    # pprint(dev_out)


def get_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dexp', help='root experiment folder', default='exp')
    parser.add_argument('--model', help='which model to use', default='glad', choices=get_models())
    parser.add_argument('--epoch', help='max epoch to run for', default=100, type=int)
    parser.add_argument('--demb', help='word embedding size', default=400, type=int)
    parser.add_argument('--dhid', help='hidden state size', default=200, type=int)
    parser.add_argument('--batch_size', help='batch size', default=64, type=int)  #50
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--stop', help='slot to early stop on', default='joint_goal')
    parser.add_argument('--resume', help='save directory to resume from')
    parser.add_argument('-n', '--nick', help='nickname for model', default='test')
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    parser.add_argument('--test', action='store_true', help='run in evaluation only mode')
    parser.add_argument('--gpu', type=int, help='which GPU to use')
    parser.add_argument('--dropout', nargs='*', help='dropout rates', default=['emb=0.2', 'local=0.2', 'global=0.2'])
    # parser.add_argument('--dann', default="data/demo_add/ann")
    parser.add_argument('--data_type', default="dstc")

    parser.add_argument('--dann_woz', default="data/woz/ann")
    parser.add_argument('--dann_dstc', default="data/DSTC/dstc/ann")

    TRY = False
    if TRY:
        parser.set_defaults(dann_woz="data/demo_add/ann")
        parser.set_defaults(dann_dstc="data/DSTC/demo_dstc/ann")
        # parser.set_defaults(dann="data/DSTC/demo_dstc/ann")
        # arg_parser.set_defaults(valid_path="data\datasets\demo_conll04\conll04_dev.json")
        parser.set_defaults(epoch=1)

    args = parser.parse_args()
    args.dout = os.path.join(args.dexp, args.model, args.nick)
    args.dropout = {d.split('=')[0]: float(d.split('=')[1]) for d in args.dropout}
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    print("------Begin time is %s------"%( time.asctime(time.localtime(time.time())) ))
    begin_time = time.time()
    args = get_args()
    run(args)
    print("------End time is %s. Total cost time %.3f min------"%(time.asctime( time.localtime(time.time()) ), (time.time()-begin_time)/60))
