import numpy as np


def linear_map(value, minimum, maximum, ranges):
    value = (value - minimum) / (maximum - minimum)

    length = len(ranges) - 1
    back = int(np.floor(value * length))
    front = int(np.ceil(value * length))

    w_back = 1.0 - (value * length - back)
    w_front = 1.0 - (front - value * length)

    if w_back == 1.0 and w_front == 1.0:
        w_front = 0.0

    out = ranges[back] * w_back + ranges[front] * w_front
    return tuple(out * (maximum - minimum) + minimum)


def heat_map(value, minimum, maximum):
    return linear_map(value, minimum, maximum,
                      np.array(
                          [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def apply_mapping(x, mapping):
    out = np.vectorize(mapping)(x)
    out = np.transpose(np.array(out), axes=[1, 2, 0])
    return out
