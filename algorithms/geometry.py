import tensorflow as tf
import numpy as np


@tf.contrib.eager.defun
def euler_to_rot(angle, config='xyz'):
    angle = tf.reshape(angle, shape=[-1, 3])
    x, y, z = tf.split(angle, 3, axis=1)
    eyes = tf.eye(3, batch_shape=[tf.shape(angle)[0]], dtype=angle.dtype)

    rots = dict()
    rots['z'] = axis_rot(z, axis='z')
    rots['y'] = axis_rot(y, axis='y')
    rots['x'] = axis_rot(x, axis='x')

    result = eyes
    for angle in list(config[::-1]):
        result = tf.matmul(result, rots[angle])

    return result


@tf.contrib.eager.defun
def to_4x4(mat):
    mat = tf.reshape(mat, shape=[-1, 3, 3])
    zeros = tf.zeros(shape=[tf.shape(mat)[0], 3, 1], dtype=mat.dtype)
    bottom = tf.constant([[0, 0, 0, 1]], dtype=mat.dtype)
    bottom = tf.tile(tf.expand_dims(bottom, 0), [tf.shape(mat)[0], 1, 1])

    mat = tf.concat([mat, zeros], axis=-1)
    mat = tf.concat([mat, bottom], axis=1)

    return mat


@tf.contrib.eager.defun
def to_rigid(r, t):
    t = tf.reshape(t, shape=[-1, 3, 1])
    r = tf.reshape(r, shape=[-1, 3, 3])
    zeros = tf.zeros(shape=[tf.shape(r)[0], tf.shape(r)[2]], dtype=r.dtype)
    ones = tf.ones(shape=[tf.shape(r)[0], 1], dtype=r.dtype)
    bottom = tf.concat([zeros, ones], axis=-1)
    bottom = tf.reshape(bottom, shape=[-1, 1, 4])

    result = tf.concat([r, t], axis=-1)
    result = tf.concat([result, bottom], axis=1)
    return result


@tf.contrib.eager.defun
def axis_rot(angle, axis='z'):
    def _z_matrix(angle):
        c = tf.cos(angle)
        s = tf.sin(angle)
        indices = tf.constant([[0], [1], [3], [4], [8]], dtype=tf.int32)
        updates = tf.stack([c, -s, s, c, 1])
        matrix = tf.scatter_nd(indices=indices, updates=updates, shape=[9])
        return tf.reshape(matrix, shape=[3, 3])

    def _y_matrix(angle):
        c = tf.cos(angle)
        s = tf.sin(angle)
        indices = tf.constant([[0], [2], [4], [6], [8]], dtype=tf.int32)
        updates = tf.stack([c, s, 1, -s, c])
        matrix = tf.scatter_nd(indices=indices, updates=updates, shape=[9])
        return tf.reshape(matrix, shape=[3, 3])

    def _x_matrix(angle):
        c = tf.cos(angle)
        s = tf.sin(angle)
        indices = tf.constant([[0], [4], [5], [7], [8]], dtype=tf.int32)
        updates = tf.stack([1, c, -s, s, c])
        matrix = tf.scatter_nd(indices=indices, updates=updates, shape=[9])
        return tf.reshape(matrix, shape=[3, 3])

    def _matrix_selector(axis):
        if axis == 'z':
            return _z_matrix
        elif axis == 'y':
            return _y_matrix
        else:
            return _x_matrix

    selected_matrix_type = _matrix_selector(axis=axis)

    return tf.map_fn(lambda x: selected_matrix_type(x[0]), angle, dtype=angle.dtype)


@tf.contrib.eager.defun
def distance_delta(x, y):
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1))


@tf.contrib.eager.defun
def rotation_delta(x, y):
    r = tf.matmul(x, tf.transpose(y, perm=[0, 2, 1]))
    trace = tf.trace(r)
    arg = tf.clip_by_value((trace - 1) / 2, clip_value_min=-0.99999, clip_value_max=0.99999)
    angle = tf.abs(tf.acos(arg))
    return angle


@tf.contrib.eager.defun
def angle_delta(x, y):
    angle_difference = tf.subtract(x, y)
    mod = tf.mod(angle_difference, tf.constant(2 * np.pi, dtype=x.dtype))
    return tf.minimum(tf.constant([2 * np.pi], dtype=x.dtype) - mod, mod)


def deg2rad(x):
    x = x * np.pi / 180.0
    x = tf.mod(x, tf.constant(2 * np.pi, dtype=x.dtype))
    x = tf.minimum(x, 2 * np.pi - x) * tf.sign(np.pi - x)

    return x


@tf.contrib.eager.defun
def skew_symmetric(x):
    matrix = tf.tile(x, tf.constant([3]))
    matrix = tf.reshape(matrix, shape=[3, 3])
    return tf.cross(tf.eye(3, dtype=x.dtype), matrix)


@tf.contrib.eager.defun
def log_map(x):
    theta = tf.acos((tf.trace(x) - 1) / 2)

    condition = tf.less(theta, 1e-5)

    coeff = tf.where(condition, tf.ones_like(theta), theta / tf.sin(theta) / 2)

    skew = tf.multiply(tf.reshape(coeff, shape=(-1, 1, 1)), (x - tf.transpose(x, perm=[0, 2, 1])))
    skew = tf.multiply(skew, [[0, 0, 1],
                              [1, 0, 0],
                              [0, 1, 0]])
    params = tf.reduce_sum(skew, axis=1)
    c, a, b = tf.split(params, 3, axis=1)
    params = tf.concat([a, b, c], axis=-1)
    return params


@tf.contrib.eager.defun
def exp_map(x):
    thetas = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
    skews = tf.map_fn(lambda x: skew_symmetric(x), x, dtype=x.dtype)

    def _low_theta(theta, skew):
        map = tf.eye(3, dtype=x.dtype)
        map = map + tf.scalar_mul(1 + tf.square(theta) / 6 + tf.pow(theta, 4) / 120, skew)
        map = map + tf.scalar_mul(0.5 - tf.square(theta) / 24 + tf.pow(theta, 4) / 720, tf.matmul(skew, skew))
        return map

    def _high_theta(theta, skew):
        map = tf.eye(3, dtype=x.dtype)
        map = map + tf.scalar_mul(tf.sin(theta) / theta, skew)
        map = map + tf.scalar_mul((1 - tf.cos(theta)) / tf.square(theta), tf.matmul(skew, skew))
        return map

    maps = tf.map_fn(
        lambda x: tf.cond(tf.less(x[0], 1e-5), lambda: _low_theta(x[0], x[1]), lambda: _high_theta(x[0], x[1])),
        (thetas, skews), dtype=x.dtype)
    return maps
