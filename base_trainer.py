import os
import tensorflow as tf
from vocab import TF2Tokenizer
from data_feed import read_corpus, batch_yield, single_yield
from base_model import TF2BertModel
from losses import MaskedSparseCategoricalCrossEntropy

'''data'''
train_corpus_path = 'data/train.txt'
dev_corpus_path = 'data/dev.txt'

'''configs'''
LR = 0.0001
output_dim = 9
batch_size = 16
max_seq_len = 128
vocab = TF2Tokenizer()
# model_dir = 'models/alBERT_only_NER'
model_dir = 'models/test'
'''generator'''
# train_data = read_corpus(train_corpus_path)
# dev_data = read_corpus(dev_corpus_path)
# train_generator = batch_yield(train_data, batch_size=batch_size,
#                               vocab=vocab, max_seq_len=max_seq_len)
# dev_generator = batch_yield(dev_data, batch_size=batch_size,
#       vocab=vocab, max_seq_len=max_seq_len)

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

'''model'''
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = TF2BertModel(output_dim=output_dim, max_seq_len=max_seq_len,
                         model_dir=model_dir)
    model.load_pretrained('models/albert_base_zh/albert_model.ckpt')
    # loss = MaskedSparseCategoricalCrossEntropy()
    loss = 'sparse_categorical_crossentropy'
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(LR), metrics=['accuracy'])
    model.summary()

'''KERAS training'''
checkpoint_prefix = os.path.join(model_dir, "ckpt_best")
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True,
                                       save_best_only=True,
                                       monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=3, min_lr=0.00001)
]

model.fit(dataset_train, epochs=100,
          steps_per_epoch=50658 // batch_size,
          callbacks=callbacks,
          validation_data=dataset_dev,
          validation_steps=4631 // batch_size)

'''Estimator (currently not functional)'''
# keras_estimator = tf.keras.estimator.model_to_estimator(
#     keras_model=model, model_dir=model_dir, checkpoint_format='saver')
# keras_estimator.train(input_fn=input_fn, steps=25)
# eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
# print('Eval result: {}'.format(eval_result))

''' TF TRAINING '''
# with strategy.scope():
#     # Set reduction to `none` so we can do the reduction afterwards and divide by
#     # global batch size.
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#       reduction=tf.keras.losses.Reduction.NONE)
#     # or loss_fn = tf.keras.losses.sparse_categorical_crossentropy
#     def compute_loss(labels, predictions):
#         per_example_loss = loss_object(labels, predictions)
#         return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)