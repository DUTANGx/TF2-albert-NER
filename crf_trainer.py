import os
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from vocab import TF2Tokenizer
from data_feed import read_corpus, batch_yield, single_yield
from crf_model import TF2BertModel

''' DATA '''
train_corpus_path = 'data/train.txt'
dev_corpus_path = 'data/dev.txt'

''' CONFIGS '''
EPOCHS = 100
LR = 0.0001
output_dim = 9
batch_size = 32
max_seq_len = 128
train_batches = 50658
dev_batches = 4631
vocab = TF2Tokenizer()
model_dir = 'models/albert_crf'
checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

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

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

''' LOSS & METRICS '''
with strategy.scope():
    def compute_loss(labels, predictions, seq_lens, trans_param):
        per_example_loss, trans_params = tfa.text.crf_log_likelihood(
            tag_indices=labels,
            inputs=predictions,
            sequence_lengths=seq_lens,
            transition_params=trans_param)
        return tf.nn.compute_average_loss(per_example_loss,
                                          global_batch_size=batch_size), trans_params


    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    dev_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='dev_accuracy')

''' MODEL '''
with strategy.scope():
    model = TF2BertModel(output_dim=output_dim, max_seq_len=max_seq_len,
                         model_dir=model_dir)
    model.load_pretrained('models/albert_base_zh/albert_model.ckpt')
    model.summary()
    optimizer = tf.keras.optimizers.Adam(LR)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

''' TF TRAINING OPS '''
with strategy.scope():
    def train_step(inputs):
        seq, labels = inputs
        real_len = tf.math.reduce_sum(seq[:, 1], axis=-1)
        with tf.GradientTape() as tape:
            predictions = model(seq, training=True)
            loss, trans_params = compute_loss(labels, predictions, real_len,
                                              model.transition_params)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions, seq[:, 1])
        return loss


    def test_step(inputs):
        seq, labels = inputs
        real_len = tf.math.reduce_sum(seq[:, 1], axis=-1)
        predictions = model(seq, training=False)
        loss, trans_params = compute_loss(labels, predictions, real_len,
                                          model.transition_params)

        dev_loss.update_state(loss)
        dev_accuracy.update_state(labels, predictions, seq[:, 1])

''' LOOP '''
with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(
                                                          dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.experimental_run_v2(test_step, args=(dataset_inputs,))


    for epoch in range(EPOCHS):
        # print(model.transition_params)
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in tqdm(dataset_train.take(train_batches//batch_size)):
            total_loss += distributed_train_step(x)
            # total_loss += train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in tqdm(dataset_dev.take(dev_batches//batch_size)):
            distributed_test_step(x)
            # test_step(x)

        checkpoint.save(checkpoint_prefix)

        template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                    "Test Accuracy: {}")
        print(template.format(epoch + 1, train_loss,
                              train_accuracy.result() * 100, dev_loss.result(),
                              dev_accuracy.result() * 100))

        dev_loss.reset_states()
        train_accuracy.reset_states()
        dev_accuracy.reset_states()