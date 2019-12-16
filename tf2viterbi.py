import numpy as np
import tensorflow as tf


# @tf.function
def viterbi_decode(score, transition_params, real_len):
    trellis = tf.Variable(tf.zeros_like(score))
    backpointers = tf.Variable(tf.zeros_like(score, dtype=tf.int32))
    trellis[0].assign(score[0])
    for t in range(1, real_len):
        v = tf.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t].assign(score[t] + tf.keras.backend.max(v, 0))
        backpointers[t].assign(
            tf.math.argmax(v, 0, output_type=tf.dtypes.int32))

    viterbi = [tf.math.argmax(trellis[-1], output_type=tf.dtypes.int32)]
    for bp in tf.reverse(backpointers[1:], [0]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = tf.keras.backend.max(trellis[-1])
    return tf.stack(viterbi), viterbi_score
    # return viterbi, viterbi_score


a = tf.convert_to_tensor(np.random.rand(5, 3))
b = tf.convert_to_tensor(np.random.rand(3, 3))
print(viterbi_decode(a, b, 5)[0].numpy())

