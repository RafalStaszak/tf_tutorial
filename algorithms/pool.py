import numpy as np
import tensorflow as tf


class TensorPool:

    def __init__(self, init_value) -> None:
        self._pool = init_value
        self._count = tf.shape(init_value)[0]

    def push(self, x):
        count = tf.shape(x)[0]
        indices = tf.random.uniform(minval=0, maxval=self._count, shape=[count], dtype=tf.int32)
        indices = tf.expand_dims(indices, 1)
        tf.scatter_nd_update(self._pool, indices, x)

    @property
    def get(self):
        return self._pool


class RandomPool:
    def __init__(self, size) -> None:
        self.data = []
        self.size = size

    def push(self, item):
        if self.count < self.size:
            self.data.append(item)
        else:
            idx = np.random.randint(self.count)
            self.data[idx] = item

    def pop(self):
        if self.count == self.size:
            idx = np.random.randint(self.count)
            return self.data.pop(idx)
        else:
            idx = np.random.randint(self.count)
            return self.data[idx]

    @property
    def count(self):
        return len(self.data)


def nhwc(n, h, w, c):
    return np.array([n, h, w, c], dtype=np.int32)


def hwc(h, w, c):
    return np.array([-1, h, w, c], dtype=np.int32)


def hw(h, w):
    return np.array([h, w], dtype=np.int32)
