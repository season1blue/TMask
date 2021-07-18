import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
import logging
import os
import re
import json
from collections import defaultdict
from copy import deepcopy
from pprint import pformat
import random

def pad(seqs, emb, device, pad=0):
    lens = [len(s) for s in seqs]
    max_len = max(lens)
    padded = torch.LongTensor([s + (max_len-l) * [pad] for s, l in zip(seqs, lens)])
    return emb(padded.to(device)), lens


def run_rnn(rnn, inputs, lens):
    # sort by lens
    order = np.argsort(lens)[::-1].tolist()
    reindexed = inputs.index_select(0, inputs.data.new(order).long())
    reindexed_lens = [lens[i] for i in order]
    packed = nn.utils.rnn.pack_padded_sequence(reindexed, reindexed_lens, batch_first=True)
    outputs, _ = rnn(packed)
    padded, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True, padding_value=0.)
    reverse_order = np.argsort(order).tolist()
    recovered = padded.index_select(0, inputs.data.new(reverse_order).long())
    # reindexed_lens = [lens[i] for i in order]
    # recovered_lens = [reindexed_lens[i] for i in reverse_order]
    # assert recovered_lens == lens
    return recovered


def attend(seq, cond, lens):
    """
    attend over the sequences `seq` using the condition `cond`.
    """
    scores = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
    return context, scores


class FixedEmbedding(nn.Embedding):
    """
    this is the same as `nn.Embedding` but detaches the result from the graph and has dropout after lookup.
    """

    def __init__(self, *args, dropout=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        out.detach_()
        return F.dropout(out, self.dropout, self.training)


class SelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -np.inf
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context


class GLADEncoder(nn.Module):
    """
    the GLAD encoder described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, din, dhid, slots, dropout=None):
        super().__init__()
        self.dropout = dropout or {}
        self.global_rnn = nn.LSTM(din, dhid, bidirectional=True, batch_first=True)
        self.global_selfattn = SelfAttention(2 * dhid, dropout=self.dropout.get('selfattn', 0.))
        for s in slots:
            setattr(self, '{}_rnn'.format(s), nn.LSTM(din, dhid, bidirectional=True, batch_first=True, dropout=self.dropout.get('rnn', 0.)))
            setattr(self, '{}_selfattn'.format(s), SelfAttention(2*dhid, dropout=self.dropout.get('selfattn', 0.)))
        self.slots = slots
        self.beta_raw = nn.Parameter(torch.Tensor(len(slots)))
        nn.init.uniform_(self.beta_raw, -0.01, 0.01)

    def beta(self, slot):
        return F.sigmoid(self.beta_raw[self.slots.index(slot)])





    def forward(self, x, x_len, slot, default_dropout=0.2):
        local_rnn = getattr(self, '{}_rnn'.format(slot))
        local_selfattn = getattr(self, '{}_selfattn'.format(slot))
        beta = self.beta(slot)
        local_h = run_rnn(local_rnn, x, x_len)
        global_h = run_rnn(self.global_rnn, x, x_len)
        h = F.dropout(local_h, self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(global_h, self.dropout.get('global', default_dropout), self.training) * (1-beta)
        c = F.dropout(local_selfattn(h, x_len), self.dropout.get('local', default_dropout), self.training) * beta + F.dropout(self.global_selfattn(h, x_len), self.dropout.get('global', default_dropout), self.training) * (1-beta)
        return h, c

from transformers import BertTokenizer
tokenizer_path = "bert/"
lower_case = False
cache_path = "data/cache"

class Model(nn.Module):
    """
    the GLAD model described in https://arxiv.org/abs/1805.09655.
    """

    def __init__(self, args, ontology, vocab, des):
        super().__init__()
        self.optimizer = None
        self.args = args
        self.vocab = vocab
        self.ontology = ontology
        self.emb_fixed = FixedEmbedding(len(vocab), args.demb, dropout=args.dropout.get('emb', 0.2))

        self.utt_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.act_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.ont_encoder = GLADEncoder(args.demb, args.dhid, self.ontology.slots, dropout=args.dropout)
        self.utt_scorer = nn.Linear(2 * args.dhid, 1)
        self.score_weight = nn.Parameter(torch.Tensor([0.5]))
        self.new_weight = nn.Parameter(torch.Tensor([0.5]))

        self.news_scorer = nn.Linear(2*args.dhid, 1)

        # self.tokenizer =BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=lower_case, cache_dir=cache_path)
        drop_parameter = 0.1
        mask_types = 2
        self.dropout = nn.Dropout(drop_parameter)
        self.mask_classifier = nn.Linear(args.demb, mask_types)

    @property
    def device(self):
        if self.args.gpu is not None and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.args.lr)

    def load_emb(self, Eword):
        new = self.emb_fixed.weight.data.new
        self.emb_fixed.weight.data.copy_(new(Eword))



    def mask_pred(self, num_transcript, utterance, utterance_len, mask, emb):
        batch_size = len(mask)
        # print(mask)
        mask_clf_list = []
        for bs in range(batch_size):
            # curr_nt = torch.LongTensor(num_transcript[bs])
            curr_len = utterance_len[bs]
            curr_utt = utterance[bs][:curr_len,:]
            curr_mask = torch.LongTensor(mask[bs]).to(self.device).unsqueeze(0)      #[1, 7, 6]
            # emb_nt = emb(curr_nt.to(self.device))       #[6, 400]

            # 传入了被GladEncoder编码的utterance 根据len进行切割 去掉padding的部分
            m = (curr_mask.unsqueeze(-1) == 0 ).float() * (-1e30)  #[1, 7, 6, 1] 7种mask的可能性 6的长度
            t = curr_utt.unsqueeze(0).repeat(1, curr_mask.shape[1], 1, 1)
            entity_span_pool = m + t  # 1 7 6 400
            entity_span_pool = entity_span_pool.max(dim=2)[0]  #[1, 7, 400]
            entity_repr = self.dropout(entity_span_pool)
            mask_clf = self.mask_classifier(entity_repr.squeeze(0))
            mask_clf = torch.softmax(mask_clf, dim=1)
            mask_clf_list.append(mask_clf)
            # print(entity_span_pool.size())
        return mask_clf_list

    def mask_pred_multi(self, num_transcript, utterance, utterance_len, mask, emb):
        batch_size = len(mask)
        # print(mask)
        mask_clf_list = []
        for bs in range(batch_size):
            # curr_nt = torch.LongTensor(num_transcript[bs])
            curr_len = utterance_len[bs]
            curr_utt = utterance[bs][:curr_len,:]   #   [6, 400]
            curr_mask = torch.LongTensor(mask[bs]).to(self.device)      #[7, 6]

            # 传入了被GladEncoder编码的utterance 根据len进行切割 去掉padding的部分
            curr_mask = curr_mask.float()
            m = torch.matmul(curr_mask, curr_utt)
            entity_repr = self.dropout(m)
            mask_clf = self.mask_classifier(entity_repr)
            mask_clf = torch.softmax(mask_clf, dim=1)
            mask_clf_list.append(mask_clf)
        return mask_clf_list

    def gen_true_mask(self, mask, pos):
        # mask是整体mask pos和neg都有     而pos只是pos
        batch_size = len(mask)
        mask_label = []
        for bs in range(batch_size):
            curr_mask = mask[bs]
            curr_pos = pos[bs]
            m_label = [[False, True] for i in range(len(mask[bs]))]  #第一列mask为True 第二列mask为False
            for p in curr_pos:
                true_index = curr_mask.index(p)
                m_label[true_index][0] = True
                m_label[true_index][1] = False
            m_label = torch.LongTensor(m_label).to(self.device).float()
            mask_label.append(m_label)
            # print(m_label)  #如果是未打乱之前的 那么应该只有前几个为True 后面全是False
        return mask_label



    def get_pred_mask(self, utterance, mask_clf, entity_mask, threshold = 0.4):
        entity_mask_tensor_list = []
        for em in entity_mask:
            entity_mask_tensor_list.append(torch.LongTensor(em).to(self.device))
        weight = []
        for mc, em in zip(mask_clf, entity_mask_tensor_list):
            pos_clf = mc[:,0]>threshold
            weighted_mask_list = []
            for mask_weight, mask in zip(pos_clf, em):
                if mask_weight:
                    weighted_mask_list.append(mask)
            if len(weighted_mask_list)==0:
                return None
            weighted_mask_tensor = torch.stack(weighted_mask_list)
            weighted_mask_tensor = torch.sum(weighted_mask_tensor, dim=0)
            weight.append(weighted_mask_tensor)

        # pad
        max_weight_len = max([w.size(0) for w in weight])
        for weight_index in range(len(weight)):
            pad_num = max_weight_len - weight[weight_index].size(0)
            weight[weight_index] = torch.nn.functional.pad(weight[weight_index], pad=(0, pad_num), mode='constant',
                                                           value=0)
        weight_tensor = torch.stack(weight)
        weighted_utterance = self.make_weighted(weight_tensor, utterance)

        return weighted_utterance


    # 相乘mask纵列相加 横行归一化
    def cal_weighted_mask(self, mask_clf, entity_mask_list):
        entity_mask_tensor_list = []
        weight = []
        for em in entity_mask_list:
            entity_mask_tensor_list.append(torch.LongTensor(em).to(self.device))

        for mc, emtl in zip(mask_clf, entity_mask_tensor_list):
            pos_clf = mc[:,0]
            weighted_mask_list = []
            for mask_weight, mask in zip(pos_clf, emtl):
                weighted_mask_list.append(mask_weight*mask)
            weighted_mask_tensor = torch.stack(weighted_mask_list)
            weighted_mask_sum = torch.sum(weighted_mask_tensor, dim=0) #tensor([0.0000, 0.9350, 1.4798, 1.7114, 1.0551, 0.0000]
            weighted_mask_sum = F.softmax(weighted_mask_sum)  #TODO 是否有必要softmax需要思考

            # 到这一步 weight_mask_sum就可以表明predicted出的一个句子中每个token关于是否有slot这件事的概率值 也可以理解为信息量
            weight.append(weighted_mask_sum)

        # pad
        max_weight_len = max([w.size(0) for w in weight])
        for weight_index in range(len(weight)):
            pad_num = max_weight_len-weight[weight_index].size(0)
            weight[weight_index] = torch.nn.functional.pad(weight[weight_index], pad=(0,pad_num), mode='constant', value=0)
        weight_tensor = torch.stack(weight)

        return weight_tensor

    def make_weighted(self, mask_weight, thing):
        weighted_t = []
        for mw, t in zip(mask_weight, thing):  # thing 可替换成替他等长度的东西
            t_mw = t * mw.unsqueeze(0).T
            weighted_t.append(t_mw)
        weighted_thing = torch.stack(weighted_t)
        return weighted_thing


    def concat_mask(self, pos, neg):
        if not pos:
            return neg
        elif not neg:
            return pos
        else:
            return pos+neg

    def forward(self, batch):
        # convert to variables and look up embeddings
        eos = self.vocab.word2index('<eos>')
        utterance, utterance_len = pad([e.num['transcript'] for e in batch], self.emb_fixed, self.device, pad=eos)
        # num_transcript =[e.num['transcript'] for e in batch]

        entity_mask, num_transcript, pos_mask = [], [], []
        for e in batch:
            # print(e.to_dict()['mask'])
            n_transcript = e.num['transcript']
            e_mask = self.concat_mask(e.to_dict()['mask']['pos'], e.to_dict()['mask']['neg'])  #还需要标注出pos
            p_mask =  e.to_dict()['mask']['pos']
            num_transcript.append(n_transcript)
            entity_mask.append(e_mask)
            pos_mask.append(p_mask)
        # entity mask是否需要shuffle 如果shuffle就在这统一shuffle mmp 不能shuffle 因为后续需要utterance和mask对应 TODO 除非弄个class 把这两个集成为一个对象中的两个属性
        # random.shuffle(entity_mask)
        # entity_mask_tensor = torch.LongTensor(entity_mask)  # cannot convert to tensor because every length of mask varies
        # print([e.to_dict()['mask'] for e in batch])


        acts = [pad(e.num['system_acts'], self.emb_fixed, self.device, pad=eos) for e in batch]
        # slot-value pairs under consideration
        ontology = {s: pad(v, self.emb_fixed, self.device, pad=eos) for s, v in self.ontology.num.items()}
        ys, exist = {}, {} # print(ontology['area'])  #'area', 'price range', 'request'
        # mask_clf_list[s] = self.mask_pred_multi(num_transcript, utterance, utterance_len, entity_mask, self.emb_fixed)
        mask_clf_list = self.mask_pred(num_transcript, utterance, utterance_len, entity_mask, self.emb_fixed)
        true_mask_list = self.gen_true_mask(entity_mask, pos_mask)
        # mask_weight = self.cal_weighted_mask(mask_clf_list, entity_mask)
        # weight_utterance = self.make_weighted(mask_weight, utterance)

        # weighted_utt = self.get_pred_mask(utterance, mask_clf_list, entity_mask)
        # if weighted_utt is not None:
        #     utterance = torch.cat([utterance, weighted_utt], dim=1)   #expand 7,25,400 ->7,50,400

        for s in self.ontology.slots:
            # for each slot, compute the scores for each value
            H_utt, c_utt = self.utt_encoder(utterance, utterance_len, slot=s)  #Hutt 7 25 400 cutt7 400
            H_acts, C_acts = list(zip(*[self.act_encoder(a, a_len, slot=s) for a, a_len in acts]))  #Hacts 7*[1 3 400]
            _, C_vals = self.ont_encoder(ontology[s][0], ontology[s][1], slot=s)  #0 represetn seqs and 1 represent lens,  only encode the values. the slot is help to discriminate

            # think add position information here  # input: e.num['transcript']  utterance e['mask'](pos and neg) utterance_len
            # H_utt = self.make_weighted(mask_weight, H_utt) #给处理之后的Hutt施加以weight

            # compute the utterance score
            y_utts, q_utts, q_acts, q_news = [], [], [], []
            for c_val in C_vals:
                cond = c_val.unsqueeze(0).expand(len(batch), *c_val.size())
                q_utt, _ = attend(H_utt, cond, lens=utterance_len)  # H_utt 7,25,400    cond 7,400
                q_utts.append(q_utt)
            # print(torch.stack(q_utts, dim=1).size()) #7 2 400
            y_utts = self.utt_scorer(torch.stack(q_utts, dim=1)).squeeze(2)

            # compute the previous action score
            for i, C_act in enumerate(C_acts):
                seq = C_act.unsqueeze(0)
                cond = c_utt[i].unsqueeze(0)
                q_act, _ = attend(seq, cond, lens=[C_act.size(0)]) #act,seq 1,1,400     utt,cond 1,400
                q_acts.append(q_act)
            y_acts = torch.cat(q_acts, dim=0).mm(C_vals.transpose(0, 1))

            # val自注意力
            # for c_val in C_vals:
            #     cond = c_val.unsqueeze(0).expand(len(batch), *c_val.size())
            #     seq = cond.unsqueeze(1)
            #     q_new, _ = attend(seq, cond, lens=[seq.size(0)])
            #     q_news.append(q_new)
            # y_news = self.news_scorer(torch.stack(q_news, dim=1)).squeeze(2)

            # exist[s] = self.exist_scorer(torch.stack(q_utts, dim=1)).squeeze(2)
            # exist[s] = torch.softmax(exist[s], dim=1)
            # exist[s] = torch.index_select(exist[s], 1, torch.tensor([0]).to(self.device))

            output = y_utts + self.score_weight * y_acts
            # output = self.make_weighted(exist[s] * 2, output).squeeze(1)
            ys[s] = F.sigmoid(output)

        if self.training:
            # print(self.ontology.values) {'area': ['south', 'west'], 'price range': ['cheap', 'expensive'], 'request': ['address', 'phone']}
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    # print("%s %s %s"%(s, v, self.ontology.values[s].index(v)))  #area south 0
                    labels[s][i][self.ontology.values[s].index(v)] = 1  #牛啊 核心在s上 因为一句话虽然可能有三个turn_label 但是每种只有一个值 这样就做到了最终与utt数对齐
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}
            # exist_labels = {s: (torch.sum(labels[s],dim=1)!=0).float().unsqueeze(1) for s in labels}

            loss, mask_loss, exist_loss = 0, 0, 0
            for s in self.ontology.slots:
                loss += F.binary_cross_entropy(ys[s], labels[s])
                # exist_loss += F.binary_cross_entropy(exist[s], exist_labels[s])
            for index in range(len(true_mask_list)):
                m_loss = F.binary_cross_entropy(mask_clf_list[index], true_mask_list[index])
                mask_loss += m_loss

        else:
            loss = torch.Tensor([0]).to(self.device)
            mask_loss = torch.Tensor([0]).to(self.device)
            # exist_loss = torch.Tensor([0]).to(self.device)
        score = {s: v.data.tolist() for s, v in ys.items()}
        # mask_score = {s: v.data.tolist() for s,v in mask_clf_list.items()}
        # print(mask_score)

        return loss, score, mask_clf_list, true_mask_list

    def get_train_logger(self):
        logger = logging.getLogger('train-{}'.format(self.__class__.__name__))
        formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
        file_handler = logging.FileHandler(os.path.join(self.args.dout, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_train(self, train, dev, args):
        # print(train.to_dict())
        track = defaultdict(list)
        iteration = 0
        best = {}
        logger = self.get_train_logger()
        if self.optimizer is None:
            self.set_optimizer()

        for epoch in range(args.epoch):
            logger.info('starting epoch {}'.format(epoch))

            # train and update parameters
            self.train()
            # print("train:")  是一个batch  但是在一个batch中含有多个turn和两个dialogue
            for batch in train.batch(batch_size=args.batch_size, shuffle=True, type="train"):
                # for t in batch:
                #     print(t.to_dict())
                iteration += 1   # the batch number
                self.zero_grad()
                loss, scores, mask_score, true_mask = self.forward(batch)
                loss.backward()
                self.optimizer.step()
                track['loss'].append(loss.item())

            # evalute on train and dev
            summary = {'iteration': iteration, 'epoch': epoch}
            #   k:loss.  multi-batch. divide by the batch num
            for k, v in track.items():
                summary[k] = sum(v) / len(v)
            print("================")
            summary.update({'eval_train_{}'.format(k): v for k, v in self.run_eval(train, args).items()})
            print("----------------")
            summary.update({'eval_dev_{}'.format(k): v for k, v in self.run_eval(dev, args).items()})
            print("================")

            print("")
            # do early stopping saves
            stop_key = 'eval_dev_{}'.format(args.stop)
            train_key = 'eval_train_{}'.format(args.stop)
            if best.get(stop_key, 0) <= summary[stop_key]:
                # print(best.get(stop_key, 0))
                # print(summary[stop_key])
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                self.save(best, identifier='epoch={epoch},iter={iteration},train_{key}={train},dev_{key}={dev}'.format(epoch=epoch, iteration=iteration, train=best_train, dev=best_dev, key=args.stop, ) )
                self.prune_saves()  # detele some model that unneccesary to save
                dev.record_preds(
                    preds=self.run_pred(dev, self.args)[0],
                    to_file=os.path.join(self.args.dout, 'dev.pred.json'),
                )
            summary.update({'best_{}'.format(k): v for k, v in best.items()})

            logger.info(pformat(summary))  #INFO:train-Model
            track.clear()

    def extract_predictions(self, scores, mask_score, threshold=0.4):
        batch_size = len(list(scores.values())[0])
        predictions = [set() for i in range(batch_size)]
        for s in self.ontology.slots:
            for i, p in enumerate(scores[s]):
                triggered = [(s, v, p_v) for v, p_v in zip(self.ontology.values[s], p) if p_v > 0.3]
                if s == 'request':
                    # we can have multiple requests predictions
                    predictions[i] |= set([(s, v) for s, v, p_v in triggered])
                elif triggered:
                    # only extract the top inform prediction
                    sort = sorted(triggered, key=lambda tup: tup[-1], reverse=True)
                    predictions[i].add((sort[0][0], sort[0][1]))

        # mask是与slot种类无关的 需要结合entity_mask及mask_score判断我预测哪些mask为True 然后将预测为True的mask与pos比对
        pred_mask = [ ms>threshold for ms in mask_score]
        # print(predictions)

        return predictions, pred_mask

    # prediction and evaluate
    def run_pred(self, dev, args):
        self.eval()
        predictions = []
        mask_predictions = []
        gold_masks = []
        for batch in dev.batch(batch_size=args.batch_size, type="eval "):
            loss, scores, mask_score, gold_mask = self.forward(batch)
            pred, pred_mask = self.extract_predictions(scores, mask_score)
            predictions += pred
            mask_predictions += pred_mask
            gold_masks += gold_mask

        return predictions, mask_predictions, gold_masks

    def run_eval(self, dev, args):
        predictions, mask_predictions, gold_mask = self.run_pred(dev, args)
        return dev.evaluate_preds(predictions, mask_predictions, gold_mask)

    def save_config(self):
        fname = '{}/config.json'.format(self.args.dout)
        with open(fname, 'wt') as f:
            # logging.info('saving config to {}'.format(fname))
            json.dump(vars(self.args), f, indent=2)

    @classmethod
    def load_config(cls, fname, ontology, **kwargs):
        with open(fname) as f:
            logging.info('loading config from {}'.format(fname))
            args = object()
            for k, v in json.load(f):
                setattr(args, k, kwargs.get(k, v))
        return cls(args, ontology)

    def save(self, summary, identifier):
        fname = '{}/{}.t7'.format(self.args.dout, identifier)
        logging.info('saving model to {}'.format(fname))
        state = {
            'args': vars(self.args),
            'model': self.state_dict(),
            'summary': summary,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, fname)

    def load(self, fname):
        logging.info('loading model from {}'.format(fname))
        state = torch.load(fname)
        self.load_state_dict(state['model'])
        self.set_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])





    def get_saves(self, directory=None):
        if directory is None:
            directory = self.args.dout
        files = [f for f in os.listdir(directory) if f.endswith('.t7')]
        scores = []
        for fname in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.stop)
            dev_acc = re.findall(re_str, fname)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(directory, fname)))
        if not scores:
            raise Exception('No files found!')
        scores.sort(key=lambda tup: tup[0], reverse=True)  # what a nb! here sort the model for list according by score
        return scores

    def prune_saves(self, n_keep=5):
        scores_and_files = self.get_saves()
        # if the number of model in folder is larger than n_keep, this will delete
        if len(scores_and_files) > n_keep:
            for score, fname in scores_and_files[n_keep:]:
                try:
                    os.remove(fname)
                except:
                    print("delete save fail")



    def load_best_save(self, directory):
        if directory is None:
            directory = self.args.dout

        scores_and_files = self.get_saves(directory=directory)
        if scores_and_files:
            assert scores_and_files, 'no saves exist at {}'.format(directory)
            score, fname = scores_and_files[0]
            self.load(fname)
