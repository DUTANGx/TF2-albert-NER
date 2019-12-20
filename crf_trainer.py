import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from vocab import TF2Tokenizer
from data_feed import read_corpus, batch_yield, single_yield
from crf_model import TF2BertModel
from tf2viterbi import viterbi_decode

''' CONFIGS '''
FOLD = 20
EPOCHS = 100
LR = 1e-6
batch_size = 16

# if you want to train on MSRA
# train_corpus_path = 'data/train.txt'
# dev_corpus_path = 'data/dev.txt'
# output_dim = 9
# train_batches = 50658
# dev_batches = 4631
# max_seq_len = 128

# if you want to train on BOSON
train_corpus_path = 'data/Boson_train.json'
dev_corpus_path = 'data/Boson_dev.json'
output_dim = 15
train_batches = 10845
dev_batches = 571
max_seq_len = 256

vocab = TF2Tokenizer()
model_dir = 'models/albert_crf'
checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

'''TFDATA'''
ds_train = tf.data.Dataset.from_generator(single_yield,
                                          args=[train_corpus_path, max_seq_len],
                                          output_types=(tf.int32, tf.int32),
                                          output_shapes=(
                                              [2, max_seq_len], [max_seq_len]))
dataset_train = ds_train.repeat().batch(batch_size)
ds_dev = tf.data.Dataset.from_generator(single_yield,
                                        args=[dev_corpus_path, max_seq_len],
                                        output_types=(tf.int32, tf.int32),
                                        output_shapes=(
                                            [2, max_seq_len], [max_seq_len]))
dataset_dev = ds_dev.repeat().batch(batch_size)

''' LOSS & METRICS '''


# with strategy.scope():
def compute_loss(labels, predictions, seq_lens, trans_param):
    per_example_loss, trans_params = tfa.text.crf_log_likelihood(
        tag_indices=labels,
        inputs=predictions,
        sequence_lengths=seq_lens,
        transition_params=trans_param)
    # return tf.nn.compute_average_loss(per_example_loss,
    #                                   global_batch_size=batch_size), trans_params
    return -tf.reduce_mean(per_example_loss), trans_params


dev_loss = tf.keras.metrics.Mean(name='dev_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='dev_accuracy')
train_F1 = tfa.metrics.F1Score(num_classes=output_dim, average='weighted')
dev_F1 = tfa.metrics.F1Score(num_classes=output_dim, average='weighted')

''' MODEL '''
model = TF2BertModel(output_dim=output_dim, max_seq_len=max_seq_len,
                     model_dir=model_dir)
# model.load()
model.load_pretrained('models/albert_base_zh/albert_model.ckpt')
model.summary()
optimizer = tf.keras.optimizers.Adam(LR)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

''' TF TRAINING OPS '''


def predict_and_record(labels, predictions, real_len, trans_params,
                       recorders, batch_size):
    for i in range(batch_size):
        # viterbi_seq, _ = viterbi_decode(
        #     predictions[i, :real_len[i]],
        #     trans_params, real_len[i])
        viterbi_seq, _ = tfa.text.viterbi_decode(
            predictions[i, :real_len[i]],
            trans_params)
        # print(labels[i, :real_len[i]], viterbi_seq)
        # print(tf.one_hot(viterbi_seq))
        for recorder in recorders:
            if isinstance(recorder, tf.keras.metrics.SparseCategoricalAccuracy):
                recorder.update_state(labels[i, :real_len[i]],
                                      tf.one_hot(viterbi_seq, model.output_dim))
            elif isinstance(recorder, tfa.metrics.F1Score):
                recorder.update_state(
                    tf.one_hot(labels[i, :real_len[i]], model.output_dim),
                    tf.one_hot(viterbi_seq, model.output_dim))


def train_step(inputs, update_recorder=False):
    seq, labels = inputs
    real_len = tf.math.reduce_sum(seq[:, 1], axis=-1)
    with tf.GradientTape() as tape:
        predictions = model(seq, training=True)
        loss, transition_params = compute_loss(labels, predictions, real_len,
                                               model.transition_params)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if update_recorder:
        predict_and_record(labels, predictions, real_len,
                           model.transition_params,
                           [train_accuracy, train_F1], batch_size)
    # train_accuracy.update_state(labels, tf.nn.softmax(predictions), seq[:, 1])

    return loss


def test_step(inputs):
    seq, labels = inputs
    real_len = tf.math.reduce_sum(seq[:, 1], axis=-1)
    predictions = model(seq, training=False)
    loss, trans_params = compute_loss(labels, predictions, real_len,
                                      model.transition_params)
    dev_loss.update_state(loss)
    predict_and_record(labels, predictions, real_len, trans_params,
                       [dev_accuracy, dev_F1], batch_size)
    # dev_accuracy.update_state(labels, tf.nn.softmax(predictions),
    #                             seq[:, 1])


''' LOOP '''

for epoch in range(EPOCHS):
    # print(model.transition_params)
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for _ in range(FOLD):
        for x in tqdm(dataset_train.take(train_batches // batch_size // FOLD)):
            # total_loss += distributed_train_step(x)
            total_loss += train_step(x, False)
            num_batches += 1
        for x in dataset_train.take(1):
            total_loss += train_step(x, True)
            num_batches += 1
        train_loss = total_loss / num_batches
        template = ("Loss: {}, Accuracy: {}% F1: {}%")
        print(template.format(train_loss,
                              train_accuracy.result() * 100,
                              train_F1.result()))
    # TEST LOOP
    for x in tqdm(dataset_dev.take(dev_batches // batch_size)):
        # distributed_test_step(x)
        test_step(x)

    checkpoint.save(checkpoint_prefix)

    template = ("Epoch {}, Loss: {}, Accuracy: {}%, F1: {}% Test Loss: {}, "
                "Test Accuracy: {}% TestF1: {}%")
    print(template.format(epoch + 1, train_loss,
                          train_F1.result() * 100,
                          train_accuracy.result() * 100,
                          dev_loss.result(),
                          dev_accuracy.result() * 100,
                          dev_F1.result()))

    dev_loss.reset_states()
    train_accuracy.reset_states()
    dev_accuracy.reset_states()
