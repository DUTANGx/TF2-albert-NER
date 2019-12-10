import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K


class MaskedSparseCategoricalCrossEntropy(keras.losses.Loss):
    def call(self, label, y_pred):
        loss = K.sparse_categorical_crossentropy(label,
                                                 y_pred) * tf.dtypes.cast(
            tf.less(label, 9),
            tf.dtypes.float32)
        loss *= tf.dtypes.cast(tf.size(loss), tf.dtypes.float32)
        loss /= K.sum(tf.dtypes.cast(tf.less(label, 9), tf.dtypes.float32))
        return loss

def masked_sparse_categorical_crossentropy(label, y_pred):
    # label1 = tf.dtypes.cast(label * tf.less(label, 9), tf.dtypes.int32)
    # mask2 = tf.dtypes.cast(mask1, tf.dtypes.float32)
    loss = K.sparse_categorical_crossentropy(label, y_pred) * tf.dtypes.cast(
        tf.less(label, 9),
        tf.dtypes.float32)
    loss *= tf.dtypes.cast(tf.size(loss), tf.dtypes.float32)
    loss /= K.sum(tf.dtypes.cast(tf.less(label, 9), tf.dtypes.float32))
    return loss


# loss = MaskedSparseCategoricalCrossEntropy()
y_true = tf.convert_to_tensor([[0, 8, 9]])
y_pred = tf.convert_to_tensor(
    [[[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]])
masked_sparse_categorical_crossentropy(y_true, y_pred)
