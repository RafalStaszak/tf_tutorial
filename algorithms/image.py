import tensorflow as tf
import numpy as np


def pixelwise_mean(x, kernel_size=3):
    return tf.layers.average_pooling2d(x, pool_size=kernel_size, strides=1, padding='SAME')


def pixelwise_variance(x, kernel_size=3):
    result = pixelwise_mean(tf.square(x), kernel_size) - tf.square(pixelwise_mean(x, kernel_size))
    return result


def pixelwise_covariance(x, y, kernel_size=3):
    mean_x = pixelwise_mean(x, kernel_size)
    mean_y = pixelwise_mean(x, kernel_size)
    return tf.multiply(x - mean_x, y - mean_y)


def structural_similarity(x, y, kernel_size=3, c1=0.01, c2=0.02):
    mean_x = pixelwise_mean(x, kernel_size)
    mean_y = pixelwise_mean(y, kernel_size)
    variance_x = pixelwise_variance(x, kernel_size)
    variance_y = pixelwise_variance(y, kernel_size)
    cov = pixelwise_covariance(x, y)
    nom = (2 * mean_x * mean_y + c1) * (2 * cov + c2)
    denom = (tf.square(mean_x) + tf.square(mean_y) + c1) * (variance_x + variance_y + c2)
    return nom / denom


def coordinates(shape):
    i = shape[0]
    j = shape[1]

    x = tf.tile(tf.range(j), multiples=[i])
    x = tf.reshape(x, shape=[i, j])

    y = tf.tile(tf.range(i), multiples=[j])
    y = tf.reshape(y, shape=[j, i])

    x = tf.stack((x, tf.transpose(y, perm=[1, 0])), -1)

    return x


def non_zero_idx(x):
    i = tf.shape(x)[0]
    j = tf.shape(x)[1]

    coords = coordinates(tf.shape(x))

    x_doubled = tf.stack((x, x), -1)

    idx = tf.where(tf.cast(x_doubled, tf.bool), coords, 2 * tf.maximum(i, j) * tf.ones((i, j, 2), coords.dtype))

    return idx


def euclidean_distance_transform(x):
    idx = non_zero_idx(x)
    coords = coordinates(tf.shape(x))
    i = tf.shape(x)[0]
    j = tf.shape(x)[1]

    coords = tf.reshape(coords, shape=[i, j, 1, 2])
    idx = tf.reshape(idx, shape=[1, 1, -1, 2])
    out = tf.abs(idx - coords)
    out = tf.reduce_min(tf.reduce_sum(tf.square(out), axis=3), axis=2)

    return tf.cast(out, x.dtype)


def iou(x, y):
    shape = tf.shape(x)[-3:]

    x = tf.reshape(x, [-1, shape[0], shape[1], shape[2]])
    i = x * y
    u = x + y

    i = tf.sign(i)
    u = tf.sign(u)

    i = tf.reduce_sum(i, axis=[1, 2])
    u = tf.reduce_sum(u, axis=[1, 2])

    return i / u


def iou_numpy(x, y):
    shape = np.shape(x)[-3:]

    x = np.reshape(x, [-1, shape[0], shape[1], shape[2]])
    i = x * y
    u = x + y

    i = np.sign(i)
    u = np.sign(u)

    i = np.sum(i, axis=(1, 2))
    u = np.sum(u, axis=(1, 2))

    return i / u
