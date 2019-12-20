import sys, pickle, os, random, json
import numpy as np
from vocab import TF2Tokenizer

vocab = TF2Tokenizer()
'''MSRA'''
# tag2label = {"O": 0,
#              "B-PER": 1, "I-PER": 2,
#              "B-LOC": 3, "I-LOC": 4,
#              "B-ORG": 5, "I-ORG": 6,
#              "START": 7, "END": 8
#              }
'''BOSON'''
tag2label = {"O": 0,
             "B-person_name": 1, "I-person_name": 2,
             "B-location": 3, "I-location": 4,
             "B-org_name": 5, "I-org_name": 6,
             "B-time": 7, "I-time": 8,
             "B-company_name": 9, "I-company_name": 10,
             "B-product_name": 11, "I-product_name": 12,
             "START": 13, "END": 14
             }
tag_pad = 14


## tags, BIO
def tag_process(tag_seq, max_seq_len):
    if type(tag_seq[0]) is str:
        tag_seq = [tag_seq]
    tag_seq = list(map(lambda x: ["START"] + x + ["END"], tag_seq))
    tag_seq = list(map(lambda x: [tag2label[tag] for tag in x], tag_seq))
    tag_seq = list(
        map(lambda x: x + [tag_pad] * (max_seq_len - len(x)), tag_seq))
    return np.array(tag_seq)


def read_corpus(corpus_path, from_txt=False):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    if from_txt:
        data = []
        with open(corpus_path, encoding='utf-8') as fr:
            lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                c = line.strip().split()
                if len(c) == 1:
                    char = ' '
                    label = c[0]
                else:
                    [char, label] = c
                sent_.append(char)
                tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    else:
        with open(corpus_path) as f:
            data = json.load(f)

    return data


def batch_yield(data, batch_size, vocab, max_seq_len, shuffle=True):
    total_data_size = len(data)
    while True:
        if shuffle:
            random.shuffle(data)
        i = 0
        while i * batch_size < total_data_size:
            seqs, labels = zip(*data[i * batch_size:(i + 1) * batch_size])
            seqs = vocab.tokenize_to_ids(seqs, max_seq_len)
            labels = tag_process(labels, max_seq_len)
            yield np.array(seqs), np.array(labels)
            i += 1


def single_yield(data_dir, max_seq_len):
    data = read_corpus(data_dir)
    random.shuffle(data)
    for d in data:
        seq, label = d
        seq = vocab.tokenize_to_ids(seq, max_seq_len)[0]
        label = tag_process(label, max_seq_len)[0]
        yield seq, label


def divide_corpus():
    data = read_corpus('data/Boson.txt', from_txt=True)
    new_data = []
    for i, s in enumerate(data):
        if (len(s[0]) > 254) or (len(s[0]) == 0):
            continue
        else:
            new_data.append(s)

    divide = int(len(new_data) * 0.95)
    print(len(new_data) - divide)
    print(divide)
    with open('data/Boson_train.json', 'w') as f:
        json.dump(new_data[:divide], f)
    with open('data/Boson_dev.json', 'w') as f:
        json.dump(new_data[divide:], f)

divide_corpus()