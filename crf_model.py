import os
import bert
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from vocab import TF2Tokenizer


class TF2BertModel(keras.Model):
    def __init__(self, output_dim, max_seq_len=128,
                 model_dir="models/albert_crf"):
        # settings
        super(TF2BertModel, self).__init__()
        self.output_dim = output_dim
        self._max_seq_len = max_seq_len
        self.model_dir = model_dir
        # layers
        self.initialize_bert()
        self.final_fc = keras.layers.Dense(self.output_dim)
        # transition
        initializer = tf.keras.initializers.GlorotUniform()
        self._transition_params = tf.Variable(
            initializer([output_dim, output_dim]), "transitions")

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, seq_len):
        self._max_seq_len = seq_len

    @property
    def transition_params(self):
        return self._transition_params

    # @transition_params.setter
    # def transition_params(self, tensor):
    #     self.transition_params = tensor

    def initialize_bert(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    def call(self, input, training=False):
        """
        :param x: (input token ids, attention masks)
        :return:
        """
        x = self.l_bert(input[:, 0], input[:, 1])
        outs = self.final_fc(x)
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
        vocab.tokenize_to_ids(['马云，祖籍浙江省绍兴嵊州市谷来镇，后父母移居杭州，出生于浙江省杭州市，中国企业家，中国共产党党员。曾为亚洲首富、阿里巴巴集团董事局主席。'], model.max_seq_len))
    print(encoded.shape)
    print(np.argmax(encoded, axis=-1))

    # from losses import masked_sparse_categorical_crossentropy
    #
    # label = np.argmax(encoded, axis=-1)
    # label[label == 8] = 9
    # print(masked_sparse_categorical_crossentropy(tf.convert_to_tensor(label),
    #                                              tf.convert_to_tensor(encoded)))
