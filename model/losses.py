import tensorflow as tf
from tensorflow._api.v1.losses import *


def mae(x, y):
    return tf.reduce_mean(tf.abs(x - y))


def sce(x, y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y))


def mse(x, y):
    return tf.losses.mean_squared_error(x, y)
