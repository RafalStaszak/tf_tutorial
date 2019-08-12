import tensorflow as tf


def randomize(boxes, max_displacement, min_scale, max_scale):
    def _substitute_last_elem(vec, elem):
        length = tf.shape(vec)[0]
        begin = tf.slice(vec, [0], [length - 1])
        return tf.concat((begin, elem), axis=-1)

    disp_shape = _substitute_last_elem(tf.shape(boxes), elem=tf.constant([2], dtype=tf.int32))
    scale_shape = _substitute_last_elem(tf.shape(boxes), elem=tf.constant([1], dtype=tf.int32))

    displacement = tf.random.uniform(shape=disp_shape, minval=-max_displacement,
                                     maxval=max_displacement)

    disp_x, disp_y = tf.split(displacement, 2, axis=-1)
    scale = tf.random.uniform(shape=scale_shape, minval=min_scale, maxval=max_scale)

    centered = to_centered(boxes)

    x, y, w, h = tf.split(centered, 4, axis=-1)
    x = x + disp_x
    y = y + disp_y
    w = w * scale
    h = h * scale

    modified = tf.concat((x, y, w, h), axis=-1)

    out = from_centered(modified)
    return out


def to_centered(boxes):
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    x = x + w / 2
    y = y + h / 2

    return tf.concat((x, y, w, h), axis=-1)


def from_centered(boxes):
    x, y, w, h = tf.split(boxes, 4, axis=-1)

    x = x - w / 2
    y = y - h / 2

    return tf.concat((x, y, w, h), axis=-1)


def to_corners(boxes):
    x_1, y_1, w, h = tf.split(boxes, 4, axis=-1)

    x_2 = x_1 + w
    y_2 = y_1 + h

    return tf.concat((y_1, x_1, y_2, x_2), axis=-1)


def from_corners(corners):
    y_1, x_1, y_2, x_2 = tf.split(corners, 4, axis=-1)

    w = x_2 - x_1
    h = y_2 - y_1

    return tf.concat((x_1, y_1, w, h), axis=-1)


def fit_to_visible_area(boxes_corners, width, height):
    y_1, x_1, y_2, x_2 = tf.split(boxes_corners, 4, axis=-1)

    relu_x_1 = tf.nn.relu(-x_1)
    relu_x_2 = tf.nn.relu(x_2 - width)
    relu_y_1 = tf.nn.relu(-y_1)
    relu_y_2 = tf.nn.relu(y_2 - height)

    x_1 = x_1 + relu_x_1 - relu_x_2
    x_2 = x_2 + relu_x_1 - relu_x_2
    y_1 = y_1 + relu_y_1 - relu_y_2
    y_2 = y_2 + relu_y_1 - relu_y_2

    return tf.concat((y_1, x_1, y_2, x_2), axis=-1)


def norm_corners(corners, img_width, img_height):
    y_1, x_1, y_2, x_2 = tf.split(corners, 4, axis=-1)

    x_1 = x_1 / img_width
    x_2 = x_2 / img_width
    y_1 = y_1 / img_height
    y_2 = y_2 / img_height

    return tf.concat((y_1, x_1, y_2, x_2), axis=-1)


def randomize_and_fit(boxes, max_displacement, min_scale, max_scale, img_width, img_height):
    rand_boxes = randomize(boxes, max_displacement, min_scale, max_scale)
    corners = to_corners(rand_boxes)
    fitted_corners = fit_to_visible_area(corners, img_width, img_height)
    return from_corners(fitted_corners)


def norm_centers_to_boxes(boxes, centers):
    x, y, w, h = tf.split(boxes, 4, axis=-1)
    x_c, y_c = tf.split(centers, 2, axis=-1)

    norm_x = (x_c - x) / w
    norm_y = (y_c - y) / h

    return tf.concat((norm_x, norm_y), axis=-1)


def crop_and_pad_to_bounding_box(image, y, x, h, w, mode='CONSTANT'):
    height, width, channels = tf.split(tf.shape(image), 3, axis=-1)
    height = tf.reshape(height, shape=[])
    width = tf.reshape(width, shape=[])

    left = tf.maximum(-x, 0)
    right = tf.maximum(x + w - width, 0)
    up = tf.maximum(-y, 0)
    bottom = tf.maximum(y + h - height, 0)

    image = tf.pad(image, [[up, bottom], [left, right], [0, 0]], mode=mode)
    image = tf.image.crop_to_bounding_box(image, y + up, x + left, h, w)
    return image


def center_to_box(centers, size=0.03):
    out = tf.concat((centers, centers), axis=-1)
    x_min, y_min, x_max, y_max = tf.split(out, 4, axis=-1)
    out = tf.concat((y_min, x_min, y_max, x_max), axis=-1)

    box = tf.constant([-size, -size, size, size], dtype=tf.float32)
    out = out + box
    out = tf.expand_dims(out, axis=1)
    out = tf.clip_by_value(out, 0.0, 1.0)
    return out
