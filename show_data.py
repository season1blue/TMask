# data_path = "data/demo_add/raw/dev.json"
data_path = "data/woz/raw/train.json"
test = [801, 817, 820, 828, 859, 970, 1019, 1022, 1081]
import json

with open(data_path) as f:
    data = json.load(f)
    # print(data)
    count = 0
    sa = 0
    total_count = 0
    for dialogue in data:
        # if dialogue['dialogue_idx'] not in test: continue
        # print(dialogue['dialogue_idx'])
        for turn in dialogue['dialogue']:
            total_count += 1
            # PT = False
            transcript = turn['transcript']
            if turn['system_acts']: sa += 1
            turn_label = turn['turn_label']
            # PT = False
            # for v in turn_label:
            #     if v[0] == 'request': continue
            #     if v[1] == 'dontcare': continue
            #     # if turn['system_acts']: continue
            #     try:
            #         transcript.lower().index(v[1].lower())
            #     except:
            #         PT = True
            # if PT and turn['system_transcript'] :
            #     # print(turn)
            #     # print(turn['turn_label'])
            #     print(turn['system_transcript'])
            #     if turn['system_acts']: print(turn['system_acts'])
            #     print(turn['transcript'])
            #     print(turn['turn_label'])
            #     print("----------")
            #     count += 1
    print("---------------%s/%s------------"%(sa,total_count))
    # print("================================================")

# with open(data_path) as f:
#     data = json.load(f)
#     # print(data)
#     count = 0
#     for dialogue in data:
#         if dialogue['dialogue_idx'] not in test: continue
#         for turn in dialogue['dialogue']:
#             print(turn['system_transcript'])
#             print(turn['system_acts'])
#             print(turn['transcript'])
#             print(turn['turn_label'])
#             print("----------")
#         print("================================================")

