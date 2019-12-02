import os
import bert
import numpy as np
import tensorflow as tf
from tensorflow import keras
from vocab import TF2Tokenizer


class TF2BertModel(keras.Model):
    def __init__(self, output_dim, max_seq_len=128,
                 model_dir="models/albert_base_zh"):
        # settings
        super(TF2BertModel, self).__init__()
        self.output_dim = output_dim
        self._max_seq_len = max_seq_len
        self.model_dir = model_dir
        # layers
        self.initialize_bert()
        self.final_fc = keras.layers.Dense(self.output_dim)

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, seq_len):
        self._max_seq_len = seq_len

    def initialize_bert(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    def call(self, input, training=False):
        """
        :param x: (input token ids, attention masks)
        :return:
        """
        # mask = input[:, 1]
        # mask = tf.expand_dims(mask, 2)
        # mask = tf.dtypes.cast(mask, tf.dtypes.float32)
        x = self.l_bert(input[:, 0], input[:, 1])
        x = self.final_fc(x)
        outs = tf.nn.softmax(x)
        # outs = tf.nn.softmax(x) * mask
        return outs

    def load_pretrained(self, pretrained_ckpt):
        self.build(input_shape=(None, 2, self._max_seq_len))
        bert.load_albert_weights(self.l_bert, pretrained_ckpt)

    def load(self, weight_file):
        self.build(input_shape=(None, 2, self._max_seq_len))
        self.load_weights(weight_file)


if __name__ == '__main__':
    model = TF2BertModel(9, 128, 'models/alBERT_only_NER')
    model.load(os.path.join(model.model_dir, 'ckpt_best'))
    # model.load_pretrained('models/albert_base_zh/albert_model.ckpt')

    vocab = TF2Tokenizer()
    encoded = model.predict(
        vocab.tokenize_to_ids(['马化腾是腾讯科技股份有限公司的董事长。'], model.max_seq_len))
    print(np.argmax(encoded, axis=-1))
