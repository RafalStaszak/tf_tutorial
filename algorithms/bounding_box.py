import numpy as np


def center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return center_x, center_y


def scale(bbox, factor=1.0):
    _, _, w, h = bbox
    center_x, center_y = center(bbox)
    w = w * factor
    h = h * factor
    x = center_x - w / 2.0
    y = center_y - h / 2.0
    return np.array([x, y, w, h], dtype=np.int32)


def map(x, total, low=-1, high=1):
    return (high - low) / total * x + low


def extract_image(image, bbox):
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def square_centered(bbox, keep_max=True):
    x, y, w, h = bbox
    x_c = x + w / 2
    y_c = y + h / 2

    if (keep_max):
        dim = np.maximum(w, h)
    else:
        dim = np.minimum(w, h)

    x = x_c - dim / 2
    y = y_c - dim / 2

    return [int(x), int(y), dim, dim]


def get(image):
    h, w = np.shape(image)[:2]
    min_h = w - 1
    max_h = 0
    min_w = h - 1
    max_w = 0

    for i in range(len(image)):
        row = image[i]
        for j in range(len(row)):
            val = np.sum(image[i, j])
            if val != 0:
                min_h = np.minimum(min_h, i)
                max_h = np.maximum(max_h, i)
                min_w = np.minimum(min_w, j)
                max_w = np.maximum(max_w, j)

    return min_w, min_h, max_w - min_w, max_h - min_h


def norm(bbox, norm_x, norm_y):
    x, y, w, h = bbox
    return x / norm_x, y / norm_y, w / norm_x, h / norm_y


def overlap(bbox_1, bbox_2):
    x_1, y_1, w_1, h_1 = bbox_1
    x_2, y_2, w_2, h_2 = bbox_2

    x = x_1 if x_1 > x_2 else x_2
    y = y_1 if y_1 > y_2 else y_2

    x_br = x_1 + w_1 if x_1 + w_1 < x_2 + w_2 else x_2 + w_2
    y_br = y_1 + h_1 if y_1 + h_1 < y_2 + h_2 else y_2 + h_2

    w = x_br - x
    h = y_br - y

    if x < x_br and y < y_br:
        return [x, y, w, h]
    else:
        return None


def sum(bbox_1, bbox_2):
    x_1, y_1, w_1, h_1 = bbox_1
    x_2, y_2, w_2, h_2 = bbox_2

    x = x_1 if x_1 < x_2 else x_2
    y = y_1 if y_1 < y_2 else y_2

    x_br = x_1 + w_1 if x_1 + w_1 > x_2 + w_2 else x_2 + w_2
    y_br = y_1 + h_1 if y_1 + h_1 > y_2 + h_2 else y_2 + h_2

    w = x_br - x
    h = y_br - y

    return [x, y, w, h]


def contains(bbox_1, bbox_2):
    box = overlap(bbox_1, bbox_2)
    return box == bbox_2
