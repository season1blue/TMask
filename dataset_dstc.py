import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient
import re
import torch
from copy import deepcopy
# from tqdm import tqdm_notebook as tqdm

client = None

# punctuation = '!,;:?"\'、，；.?'
punctuation = ' '
def remove_punctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),' ',text).lower()
    return text

def annotate(sent):
    global client
    if client is None:
        client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    words = []
    for sent in client.annotate(sent).sentences:
        for tok in sent:
            words.append(tok.word)
    return words

def create_mask(index, text_len, slot_len=1):
    origin = [False] * (text_len+2)   #<sos> and <eos>
    begin_index = index +1          #<sos>
    end_index = min(begin_index+slot_len, text_len+1)
    origin[begin_index: end_index] = [True]*(end_index-begin_index)
    return origin


class Turn:

    def __init__(self, turn_id, transcript, turn_label, system_acts, system_transcript, mask, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.num = num or {}
        self.mask = mask

    def to_dict(self):
        return {'turn_id': self.id, 'transcript': self.transcript, 'turn_label': self.turn_label, 'system_acts': self.system_acts, 'system_transcript': self.system_transcript, 'mask':self.mask, 'num': self.num}

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())
        # NOTE: fix inconsistencies in data label
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}

        # create mask
        transcript = annotate(raw['transcript'])
        # transcript = remove_punctuation(deepcopy(raw['transcript'])).split(" ")
        value = [t[1] for t in raw['turn_label']]
        pos_mask_list = []
        fail = 0
        for v in value:
            v_sep = v.split(" ")
            if v_sep[0] == 'dontcare':  continue
            try:
                value_index = transcript.index(v_sep[0])
            except:
                # print(v_sep)
                fail += 1
                continue

            pos_mask = create_mask(value_index, len(transcript), len(v_sep))
            pos_mask_list.append(pos_mask)
        pos_mask_tensor = torch.LongTensor(pos_mask_list)

        neg_mask_list = []
        for neg_index in range(0, len(transcript)):
            for span in range(1,3):  # 0 1
                neg_mask = create_mask(neg_index, len(transcript), span)
                if neg_mask not in pos_mask_list and neg_mask not in neg_mask_list:
                    neg_mask_list.append(neg_mask)
        neg_mask_tensor = torch.LongTensor(neg_mask_list)
        mask ={"pos":pos_mask_list, "neg":neg_mask_list}

        return cls(
            turn_id=raw['turn_idx'],
            transcript=transcript,
            system_acts=system_acts,
            turn_label=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['turn_label']],
            system_transcript=raw['system_transcript'],
            mask = mask
        )

    def numericalize_(self, vocab):
        # because only encode the transcript and system_act, so here only embed the two information
        self.num['transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']], train=True)
        self.num['system_acts'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.system_acts + [['<sentinel>']]]



class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id, 'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw):
        dialogue_ann, history = [], []
        Fail = 0
        for t in raw['dialogue']:
            turn_ann = Turn.annotate_raw(t)
            dialogue_ann.append(turn_ann)

        # return cls(raw['dialogue_idx'], [Turn.annotate_raw(t) for t in raw['dialogue']])
        return cls(raw['dialogue_idx'], dialogue_ann)

class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    @classmethod
    def annotate_raw(cls, fname):
        with open(fname) as f:
            data = json.load(f)
            return cls([Dialogue.annotate_raw(d) for d in tqdm(data, ncols=80, desc="pre_process")])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)

    def extract_ontology(self):
        slots = set()
        values = defaultdict(set)
        for t in self.iter_turns():
            for s, v in t.turn_label:
                slots.add(s.lower())
                values[s].add(v.lower())
        return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})

    def batch(self, batch_size, shuffle=False, type=None):
        turns = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(turns)
        # for i in tqdm(range(0, len(turns), batch_size), ncols=80):
        #     yield turns[i:i+batch_size]

        try:
            with tqdm(range(0, len(turns), batch_size), ncols=60, desc=type) as t:
                for i in t:
                    yield turns[i:i + batch_size]
        except KeyboardInterrupt:
            t.close()
            raise
        t.close()

    def evaluate_preds(self, preds, pred_mask, gold_mask):
        request = []
        inform = []
        joint_goal = []
        mask_hit, mask_dim = 0, 0

        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i, j = 0, 0
        # for d in self.dialogues:
        #     pred_state = {}
        #     for t in d.turns:
        #         # print(t.to_dict())
        #         gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
        #         gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request'])
        #         pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
        #         pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
        #         request.append(gold_request == pred_request)
        #         inform.append(gold_inform == pred_inform)
        #
        #         mask = pred_mask[i] == gold_mask[i] #不是和t.mask pos比 而是和生成的true_mask比
        #         mask_hit += torch.sum(mask)
        #         mask_dim += mask.size(0)* mask.size(1)
        #
        #         gold_recovered = set()
        #         pred_recovered = set()
        #         for s, v in pred_inform:
        #             pred_state[s] = v
        #         for b in t.turn_label:
        #             if b[0]!='request':
        #                 gold_recovered.add(('inform', fix.get(b[0].strip(),b[0].strip()), fix.get(b[1].strip(),b[1].strip())))
        #         for s, v in pred_state.items():
        #             pred_recovered.add(('inform', s, v))
        #         joint_goal.append(gold_recovered == pred_recovered)
        #         i += 1
        # mask_acc = (mask_hit/mask_dim).item()
        # return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal), "mask_acc":mask_acc}

        for d in self.dialogues:
            pred_state = {}
            gold_state = {}
            j += 1
            for t in d.turns:
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request' and v != ''])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                mask = pred_mask[i] == gold_mask[i]  # 不是和t.mask pos比 而是和生成的true_mask比
                mask_hit += torch.sum(mask)
                mask_dim += mask.size(0) * mask.size(1)

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v

                for s, v in gold_inform:
                    gold_state[s] = v
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                for s, v in gold_state.items():
                    gold_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal)}



    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)

class Description:
    def __init__(self, des):
        self.des = des

    def annoate(self, fname):
        with open(fname) as f:
            data = json.load(f)
            print(data)
            # dataset_ann = []
            # for d in tqdm(data, ncols=80, desc="pre_process"):
            #     dialogue_ann, fail = Dialogue.annotate_raw(d)
            #     dataset_ann.append(dialogue_ann)
            # return dataset_ann
            return
            # return cls([Dialogue.annotate_raw(d) for d in tqdm(data, ncols=80, desc="pre_process")])



class Ontology:

    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def __add__(self, another):
        new_slots = sorted(list(set(self.slots + another.slots)))
        new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
        return Ontology(new_slots, new_values)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    def numericalize_(self, vocab):
        self.num = {}
        for s, vs in self.values.items():
            self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
