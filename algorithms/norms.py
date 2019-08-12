import tensorflow as tf


def min_max(x):
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x))


def map_range(x, old_min, old_max, new_min, new_max):
    # Figure out how 'wide' each range is
    old_span = old_max - old_min
    new_span = new_max - new_min

    # Convert the left range into a 0-1 range (float)
    scaled = (x - old_min) / old_span

    # Convert the 0-1 range into a value in the right range.
    return new_min + (scaled * new_span)
