import os
import bert
import numpy as np


class TF2Tokenizer(object):
    def __init__(self, model_dir="models/albert_base_zh"):
        self.tokenizer = bert.albert_tokenization.FullTokenizer(
            os.path.join(model_dir, 'vocab.txt'))

    def tokenize_to_ids(self, s, max_seq_len):
        """
        tokenize, to_ids, and padding
        :param s: strings, could be string or list of string
        :param max_seq_len: padding length
        :return:
        """
        # 2 cases: s is str or list of char
        if len(s[0]) == 1:
            s = [s]
        pred_tokens = map(lambda tok: ''.join(tok), s)
        pred_tokens = map(self.tokenizer.tokenize, pred_tokens)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
        pred_token_ids = list(
            map(self.tokenizer.convert_tokens_to_ids, pred_tokens))
        attention_mask = map(
            lambda tids: [1] * len(tids) + [0] * (max_seq_len - len(tids)),
            pred_token_ids)
        attention_mask = np.array(list(attention_mask))
        pred_token_ids = map(
            lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))
        stacked = np.stack([pred_token_ids, attention_mask], axis=1)
        return stacked
