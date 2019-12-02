import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


# class MaskedSparseCategoricalCrossEntropy(keras.losses.Loss):
#     def call(self, label, pred):
#         # mask = tf.dtypes.cast(pred[1], tf.dtypes.float32)
#         mask = tf.dtypes.cast(mask, tf.dtypes.float32)
#         # y_pred = pred[0]
#         # loss = K.sparse_categorical_crossentropy(label, y_pred) * mask
#         loss = K.sparse_categorical_crossentropy(label, pred) * mask
#         loss = K.sum(loss) / K.sum(mask)
#         return loss

def masked_sparse_categorical_crossentropy(label, y_pred):
    # label1 = tf.dtypes.cast(label * tf.less(label, 9), tf.dtypes.int32)
    # mask2 = tf.dtypes.cast(mask1, tf.dtypes.float32)
    loss = K.sparse_categorical_crossentropy(
        tf.dtypes.cast(
            label, tf.dtypes.int32) * tf.dtypes.cast(tf.less(label, 9),
                                                     tf.dtypes.int32),
        y_pred) * tf.dtypes.cast(tf.less(label, 9), tf.dtypes.float32)
    loss = K.sum(loss) / K.sum(
        tf.dtypes.cast(tf.less(label, 9), tf.dtypes.float32))
    return loss


# loss = MaskedSparseCategoricalCrossEntropy()
y_true = tf.convert_to_tensor([[0, 8, 9]])
y_pred = tf.convert_to_tensor(
    [[[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]])
masked_sparse_categorical_crossentropy(y_true, y_pred)
