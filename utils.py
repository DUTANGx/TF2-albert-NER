import numpy as np


def to_comma_seperated_array(x):
    x_shape = x.shape
    x_flatten = np.ravel(x)
    total_len = x_flatten.size
    for i in x_shape[::-1]:
        l = []
        for j in range(total_len // i):
            l.append('[' + ','.join(
                [str(y) for y in x_flatten[j * i:(j + 1) * i]]) + ']')
        x_flatten = l
        total_len = len(x_flatten)

    return x_flatten[0]

#
# x = np.random.rand(2, 3, 4)
# print(to_comma_seperated_array(x))
