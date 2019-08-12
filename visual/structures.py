import numpy as np
from algorithms.norms import map_range


def space_to_points(tensor, min_val=0.0, max_val=1.0):
    x, y, z = np.nonzero(tensor)
    shape = np.shape(tensor)

    x = map_range(x, 0.0, float(shape[2] - 1), min_val, max_val)
    y = map_range(y, 0.0, float(shape[1] - 1), min_val, max_val)
    z = map_range(z, 0.0, float(shape[0] - 1), min_val, max_val)

    return [x, y, z]
