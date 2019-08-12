import tensorflow as tf
import numpy as np


def dense_to_sparse(arr):
    idx = np.where(arr != 0.0)
    return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)


def vals_to_space(x, weights, min_val=0.0, max_val=1.0, out_shape=tf.constant([10, 10, 10])):
    x = (x - min_val) / (max_val - min_val)
    n = tf.cast(out_shape - tf.ones_like(out_shape), tf.float32)

    x = x * n
    a = tf.floor(x)
    b = tf.ceil(x)

    a_f = tf.reduce_sum(tf.square(x - a), axis=-1)
    b_f = tf.reduce_sum(tf.square(x - b), axis=-1)

    a_f = tf.where(tf.equal(a_f, 0.0), tf.ones_like(a_f), a_f)
    b_f = tf.where(tf.equal(b_f, 0.0), tf.ones_like(b_f), b_f)

    a_v = a_f / tf.maximum(a_f, b_f) * weights
    b_v = b_f / tf.maximum(a_f, b_f) * weights

    indices = tf.concat((a, b), axis=0)
    indices = tf.cast(indices, tf.int32)
    updates = tf.concat((a_v, b_v), axis=0)

    out = tf.scatter_nd(indices, updates, out_shape)
    out = tf.clip_by_value(out, 0.0, 1.0)
    return out


def space_to_maps(x, reduction=tf.reduce_max):
    yz = tf.expand_dims(reduction(x, axis=1), axis=-1)
    xz = tf.expand_dims(reduction(x, axis=2), axis=-1)
    xy = tf.expand_dims(reduction(x, axis=3), axis=-1)

    return tf.concat((yz, xz, xy), axis=-1)


def gaussian_kernel(mean=0.0, std=1.0, size=(5, 5, 5), norm='sum'):
    x = np.zeros(size)

    d = tf.distributions.Normal(mean, std)
    mid = np.array(size, dtype=np.float32) / 2 - 0.5

    indices = []
    for i, axis_size in enumerate(size):
        if len(indices) == 0:
            for j in range(axis_size):
                indices.append([j])
        else:
            prev_indices = indices
            indices = list()
            for idx in prev_indices:
                for j in range(axis_size):
                    indices.append(idx + [j])

    indices = np.array(indices)

    for idx in indices:
        x[tuple(idx)] = np.sqrt(np.sum(np.square(mid - np.array(idx, dtype=np.float32))))

    k = d.prob(x)

    if norm == 'sum':
        return k / tf.reduce_sum(k)
    elif norm == 'max':
        return k / tf.reduce_max(k)
    else:
        return k


def conv_kernel_3d(x, kernel, strides=(1, 1, 1, 1, 1)):
    return tf.nn.conv3d(x, filter=kernel, strides=strides, padding='SAME')


def conv_kernel_2d(x, kernel, strides=(1, 1, 1, 1)):
    return tf.nn.conv2d(x, filter=kernel, strides=strides, padding='SAME')


def conv_kernel_1d(x, kernel, strides=(1, 1, 1)):
    return tf.nn.conv1d(x, filter=kernel, strides=strides, padding='SAME')
