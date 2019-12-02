import sys, pickle, os, random
import numpy as np
from vocab import TF2Tokenizer

vocab = TF2Tokenizer()
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "START": 7, "END": 8
             }
tag_pad = 8

## tags, BIO
def tag_process(tag_seq, max_seq_len):
    if type(tag_seq[0]) is str:
        tag_seq = [tag_seq]
    tag_seq = list(map(lambda x: ["START"] + x + ["END"], tag_seq))
    tag_seq = list(map(lambda x: [tag2label[tag] for tag in x], tag_seq))
    tag_seq = list(map(lambda x: x + [tag_pad] * (max_seq_len - len(x)), tag_seq))
    return np.array(tag_seq)


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

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
