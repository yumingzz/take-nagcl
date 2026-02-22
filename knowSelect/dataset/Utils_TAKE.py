import sys
import os
import nltk
import codecs
from sys import *
import random
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# def bert_tokenizer():
#     t = AutoTokenizer.from_pretrained(
#         '/root/autodl-tmp/bert-base-uncased', local_files_only=True,
#         do_lower_case=True)  # do_lower_case Whether to lower case the input.
#     return t.tokenize, t.vocab, t.ids_to_tokens

from transformers import AutoTokenizer

def bert_tokenizer():
    model_path = "/root/autodl-tmp/bert-base-uncased"  # 你本地模型路径
    t = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,           # 用 fast/slow 都行，这里不强制
        local_files_only=True    # 只走本地，避免联网
    )

    # vocab2id：token -> id
    if hasattr(t, "vocab") and isinstance(t.vocab, dict):
        vocab2id = t.vocab
    else:
        vocab2id = t.get_vocab()

    # id2vocab：id -> token（兼容不同 tokenizer）
    if hasattr(t, "ids_to_tokens") and isinstance(t.ids_to_tokens, dict):
        id2vocab = t.ids_to_tokens
    else:
        id2vocab = {i: tok for tok, i in vocab2id.items()}

    # tokenize 函数：保持你原代码期望的调用方式
    return t.tokenize, vocab2id, id2vocab


def bert_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens).replace(' ##', '').strip()
    return detokenizer


def nltk_tokenizer():
    def tokenizer(sent):
        return nltk.word_tokenize(sent.lower())
    return tokenizer


def nltk_detokenizer():
    def detokenizer(tokens):
        return ' '.join(tokens)
    return detokenizer


def load_answer(file, tokenizer):
    print("load_answer")
    answer = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)

            assert len(temp) == 4,"all_previous_query_id;all_previous_query_id;all_previous_query_id	current_query_id	background_id;background_id 	response_content"
            if len(temp[0]) < 1:
                temp[0] = []
            else:
                temp[0] = temp[0].split(';')
            temp[2] = temp[2].split(';')
            temp[3] = tokenizer(temp[3])
            answer.append(temp)
    return answer


def load_passage(file, pool, tokenizer):  # background_id	background_content
    print("load_passage")
    poolset = set()
    for k in pool:
        poolset.update(pool[k])

    passage = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 1)
            assert len(temp) == 2, "load_passage"
            if temp[0] in poolset:
                passage[temp[0]] = ' [SEP] '.join([' '.join(tokenizer(sent)) for sent in nltk.sent_tokenize(temp[1])]).split(' ')  # list的形式
    print("passage:{}, poolset:{}".format(len(passage), len(poolset)))
    return passage  # {background_id1:background_content, background_id2:background_content}


def load_pool(file, topk=None):  # current_query_id Q0 background_id rank relevance_score model_name
    print("load_pool")
    pool = {}
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 6, "load_pool"
            if temp[0] not in pool:
                pool[temp[0]] = [temp[2]]  # {“current_query_id”:[background_id1]}
            else:
                pool[temp[0]].append(temp[2])  # {“current_query_id”:[background_id1,background_id2,background_id3...]}
    return pool


def load_qrel(file):
    print("load_qrel")
    qrel = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 4, "load_qrel"
            if int(temp[3]) > 0:
                qrel[temp[0]] = temp[2]  # {current_query_id:background_id1, current_query_id2:background_id2........}
    return qrel


def load_query(file, tokenizer):  # query_id	query_content
    print("load_query")
    query = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t',1)
            assert len(temp) == 2, "load_query"
            query[temp[0]] = tokenizer(temp[1])  # {1_1:[query_tokens],}
    return query


def load_split(dataset, file):
    print("[DEBUG] split_file abs =", os.path.abspath(file))

    train = set()
    dev = set()

    if dataset == "wizard_of_wikipedia":
        test_seen = set()
        test_unseen = set()
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                temp = line.strip('\n').strip('\r').split('\t')
                assert len(temp) == 2, "query_id train/dev/test_seen/test_unseen"
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
                elif temp[1] == 'test_seen':
                    test_seen.add(temp[0])
                elif temp[1] == 'test_unseen':
                    test_unseen.add(temp[0])

        return train, dev, test_seen, test_unseen

    elif dataset == "holl_e":
        test = set()
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                temp = line.strip('\n').strip('\r').split('\t')
                assert len(temp) == 2, "query_id train/dev/test"
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
                elif temp[1] == 'test':
                    test.add(temp[0])
        return train, dev, test

    elif dataset == "tiage":
        test = set()
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                temp = line.strip('\n').strip('\r').split('\t')
                assert len(temp) == 2, "query_id train/test"
                if temp[1] == 'train':
                    train.add(temp[0])
                elif temp[1] == 'test':
                    test.add(temp[0])
                elif temp[1] == 'dev':
                    dev.add(temp[0])
        print("====== LOAD_SPLIT RESULT ======")
        print("train_ids:", len(train))
        print("dev_ids:", len(dev))
        print("test_ids:", len(test))
        print("sample labels:", list(dev)[:3])  # 随便看3个
        print("===============================")
        return train, dev, test


def split_data(dataset, split_file, episodes):
    print("split_data:", dataset)
    train_episodes = list()
    dev_episodes = list()

    if dataset == "wizard_of_wikipedia":
        train, dev, test_seen, test_unseen = load_split(dataset, split_file)
        test_seen_episodes = list()
        test_unseen_episodes = list()
        for episode in episodes:
            if episode[0]['query_id'] in train:
                train_episodes.append(episode)
            elif episode[0]['query_id'] in dev:
                dev_episodes.append(episode)
            elif episode[0]['query_id'] in test_seen:
                test_seen_episodes.append(episode)
            elif episode[0]['query_id'] in test_unseen:
                test_unseen_episodes.append(episode)
        return train_episodes, dev_episodes, test_seen_episodes, test_unseen_episodes

    elif dataset == "holl_e":
        train, dev, test = load_split(dataset, split_file)
        test_episodes = list()
        for episode in episodes:
            if episode[0]['query_id'] in train:
                train_episodes.append(episode)
            elif episode[0]['query_id'] in dev:
                dev_episodes.append(episode)
            elif episode[0]['query_id'] in test:
                test_episodes.append(episode)
        return train_episodes, dev_episodes, test_episodes

    elif dataset == "tiage":
        train, dev, test = load_split(dataset, split_file)
        print("======")
        print("1",len(train))
        print("2",len(dev))
        print(len(test))
        test_episodes = list()
        for episode in episodes:
            if episode[0]['query_id'] in train:
                # print("episode",episode)
                # s = input()
                # print("train", train)
                # s = input()
                train_episodes.append(episode)
            elif episode[0]['query_id'] in dev:
                dev_episodes.append(episode)
            elif episode[0]['query_id'] in test:
                test_episodes.append(episode)
        print("====== SPLIT RESULT ======")
        # print("dialogs in split map:", len(dialog2label))
        # print("dialog label counts:", Counter(dialog2label.values()))
        print("train:", len(train_episodes))
        print("dev:", len(dev_episodes))
        print("test:", len(test_episodes))
        print("==========================")

        return train_episodes, dev_episodes, test_episodes
    # elif dataset == "tiage":
    #     train_ids, dev_ids, test_ids = load_split(dataset, split_file)

    #     # ---- 1) 把 query_id 标签聚合到 dialog 级别：test > dev > train ----
    #     from collections import defaultdict, Counter

    #     dialog_flags = defaultdict(set)  # dialog_id -> {"train"/"dev"/"test"}

    #     def dialog_of(qid: str) -> str:
    #         # qid 形如 "56_12"
    #         return qid.rsplit("_", 1)[0]

    #     for qid in train_ids:
    #         dialog_flags[dialog_of(qid)].add("train")
    #     for qid in dev_ids:
    #         dialog_flags[dialog_of(qid)].add("dev")
    #     for qid in test_ids:
    #         dialog_flags[dialog_of(qid)].add("test")

    #     dialog2label = {}
    #     for d, flags in dialog_flags.items():
    #         if "test" in flags:
    #             dialog2label[d] = "test"
    #         elif "dev" in flags:
    #             dialog2label[d] = "dev"
    #         else:
    #             dialog2label[d] = "train"

    #     # ---- 2) 用 dialog2label 去切 episodes ----
    #     test_episodes = []
    #     for episode in episodes:
    #         first_query_id = episode[0]["query_id"]  # e.g. "56_0"
    #         d = dialog_of(first_query_id)

    #         label = dialog2label.get(d, "train")  # split 里找不到就默认 train
    #         if label == "test":
    #             test_episodes.append(episode)
    #         elif label == "dev":
    #             dev_episodes.append(episode)
    #         else:
    #             train_episodes.append(episode)

        # ---- 3) 打印统计，确认切分正确 ----
        # print("====== SPLIT RESULT ======")
        # print("dialogs in split map:", len(dialog2label))
        # print("dialog label counts:", Counter(dialog2label.values()))
        # print("train:", len(train_episodes))
        # print("dev:", len(dev_episodes))
        # print("test:", len(test_episodes))
        # print("==========================")
        # s = input()

        # return train_episodes, dev_episodes, test_episodes


def load_IDLabel(file):
    print("load_ID_label")
    with open(file, 'r', encoding='utf-8') as f:
        ID_label = json.load(f)
    return ID_label


def load_default(dataset, answer_file, passage_file, pool_file, ID_file, query_file, tokenizer, topk=None, randoms=1, node_id_json=None):
    answer = load_answer(answer_file, tokenizer)  # [[all_previous_query_ids],current_query_id,[background_ids],[response_tokens]]
    pool = load_pool(pool_file, topk)  # {“current_query_id1”:[background_id1,background_id2,background_id3...]，“current_query_id2”:[background_id1,background_id2,background_id3...]}
    query = load_query(query_file, tokenizer)  # {current_query_id_1:[query_tokens],current_query_id_2:[query_tokens]}
    passage = load_passage(passage_file, pool, tokenizer)  # {background_id1:[background_tokens], [background_id2:[background_tokens]}
    ID_labels = load_IDLabel(ID_file)
    node_id_map = None
    if node_id_json is not None and os.path.exists(node_id_json):
        with open(node_id_json, 'r', encoding='utf-8') as f:
            node_id_map = json.load(f)
    average_pool = 0

    episodes = []
    ini_episode_index = "?"
    examples = []
    episode_index = []


    for i in tqdm(range(len(answer))):
        for j in range(randoms):
            c_id, q_id, knowledge_id, response = answer[i]  # c_id is a lis，q_id is string，p_id is a list，ans is a list
            ID_label = ID_labels[i]

            knowledge_pool = pool[q_id]

            average_pool += len(knowledge_pool)

            for p in knowledge_id:  # label knowledge sentence id
                if p not in knowledge_pool:
                    raise Exception("label knowledge is not in knowledge pool")

            # we want the correct knowledge to always be in index 0
            k = knowledge_pool.index(knowledge_id[0])
            if k == 0:
                pass
            else:
                knowledge_pool[0], knowledge_pool[k] = knowledge_pool[k], knowledge_pool[0]

            example = dict()
            example['context_id'] = c_id  # list ：[previous utterance]
            example['query_id'] = q_id  # string ：current query
            example['response'] = response  # list

            example['knowledge_pool'] = knowledge_pool  # list
            example['knowledge_label'] = knowledge_id
            example['ID_label'] = ID_label
            if node_id_map is not None:
                example['node_id'] = node_id_map.get(q_id, -1)

            example['answer_file'] = answer_file
            example['passage_file'] = passage_file
            example['pool_file'] = pool_file
            example['query_file'] = query_file

            current_episode_index = "_".join(q_id.split("_")[:-1])

            if current_episode_index != ini_episode_index:
                if len(examples) == 0:
                    pass
                else:
                    print("episode_index:", ini_episode_index)
                    episode_index.append(ini_episode_index)
                    episodes.append(examples)  # [[{example1},{example2},{example3}],[{example1},{example2},{example3}],...]
                    examples = []
                ini_episode_index = current_episode_index

            examples.append(example)  # [{example1},{example2},{example3}...]

            if i == (len(answer)-1):
                print("episode_index:", current_episode_index)
                episode_index.append(current_episode_index)
                episodes.append(examples)
                examples = []

    total_number_examples = sum([len(episode) for episode in episodes])
    print('total episodes:', len(episodes))
    print('total examples:', total_number_examples)
    print('the lowest length of episodes:', min([len(episode) for episode in episodes]))
    print('the maximum length of episodes:', max([len(episode) for episode in episodes]))
    print('average length of episodes:', total_number_examples/len(episodes))
    print("average knowledge pool:", average_pool / total_number_examples)

    return episodes, query, passage











