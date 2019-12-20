import os
import bert
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from vocab import TF2Tokenizer
from utils import to_comma_seperated_array


class TF2BertModel(keras.Model):
    def __init__(self, output_dim,
                 max_seq_len=128, model_dir="models/albert_crf",
                 hidden_dim=128):
        # settings
        super(TF2BertModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self._max_seq_len = max_seq_len
        self.model_dir = model_dir
        # lbert
        self.initialize_bert()
        # layers
        self.layer_norm1 = keras.layers.LayerNormalization()
        self.layer_norm2 = keras.layers.LayerNormalization()
        self.final_fc = keras.layers.Dense(self.output_dim)
        # Bilstm
        self.bilstm = keras.layers.Bidirectional(
            keras.layers.LSTM(hidden_dim, activation=keras.activations.elu,
                              return_sequences=True))
        # transition
        initializer = tf.keras.initializers.GlorotUniform()
        self.transition_params = tf.Variable(
            initializer([output_dim, output_dim]), name="transitions")
        # self.transition_params = None

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @max_seq_len.setter
    def max_seq_len(self, seq_len):
        self._max_seq_len = seq_len

    def initialize_bert(self):
        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    @tf.function
    def call(self, input, training=False):
        """
        :param x: (input token ids, attention masks)
        :return:
        """
        x = self.l_bert(input[:, 0], input[:, 1])
        x = self.layer_norm1(x)
        x = self.bilstm(x)
        x = self.layer_norm2(x)
        outs = self.final_fc(x)
        return outs

    def load_pretrained(self, pretrained_ckpt):
        # self.call(np.zeros((16, 2, self.max_seq_len)))
        self.build(input_shape=(None, 2, self._max_seq_len))
        bert.load_albert_weights(self.l_bert, pretrained_ckpt)

    def load(self):
        self.build(input_shape=(None, 2, self._max_seq_len))
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(),
                                         model=self)
        checkpoint.restore(tf.train.latest_checkpoint(
            os.path.join(self.model_dir, 'training_checkpoints')))
        # self.load_weights(weight_file)


def model_save():
    model = TF2BertModel(9, 128, 'models/albert_crf')
    model.load()
    vocab = TF2Tokenizer()
    export_path = 'models/albert_crf/saved_model/1'
    encoded = model.predict(
        vocab.tokenize_to_ids(['你好！'],
            model.max_seq_len))
    tf.saved_model.save(
        model, export_path,
        signatures=model.call.get_concrete_function(
            tf.TensorSpec(shape=[None, 2, 128], dtype=tf.int32, name="input")))
    model.save(os.path.join(model.model_dir, 'saved_model'), save_format='tf')
    np.save('models/albert_crf/transition.npy',model.transition_params.numpy())


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model_save()
    vocab = TF2Tokenizer()

    import requests
    import json

    data = vocab.tokenize_to_ids(['马云，祖籍浙江省绍兴嵊州市谷来镇，后父母移居杭州，出生于浙江省杭州市，中国企业家，中国共产党党员。曾为亚洲首富、阿里巴巴集团董事局主席。'], 128)
    req = '{"inputs": %s}' % to_comma_seperated_array(np.array(data))
    response = json.loads(requests.post(
            url='http://0.0.0.0:8501/v1/models/albert_ner:predict',
            data=req).text)['outputs']
    preds = np.array(response[0])
    transition_params = np.load('models/albert_crf/transition.npy')
    print(tfa.text.viterbi_decode(preds, transition_params))
    print()
