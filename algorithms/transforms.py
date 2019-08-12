import numpy as np
import numpy.linalg as lg


def euler_to_rot(v=np.eye(3), xyz=np.array([0, 0, 0]), config='xyz'):
    def _map_euler_to_digit(euler):
        if euler == 'x':
            return 2
        elif euler == 'y':
            return 1
        elif euler == 'z':
            return 0

        return None

    rotations = []
    rotations.append(rotation_matrix('x', xyz[0]))
    rotations.append(rotation_matrix('y', xyz[1]))
    rotations.append(rotation_matrix('z', xyz[2]))

    rotations = np.stack(rotations)
    order = np.array([_map_euler_to_digit(char) for char in config])
    rotations = rotations[order]

    for rot in rotations:
        v = np.matmul(v, rot)

    return v


def make_transform(rot, trans):
    trans = np.reshape(trans, (3, 1))
    tf = np.hstack((rot, trans))
    tf = np.vstack((tf, [0, 0, 0, 1]))
    return tf


def to_4x4(mat):
    return make_transform(mat, [0, 0, 0])


def rotation_matrix(axis, angle):
    matrix = np.eye(3)
    c = np.cos(angle)
    s = np.sin(angle)
    if axis == 'x':
        matrix = [[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]]
    elif axis == 'y':
        matrix = [[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]]
    elif axis == 'z':
        matrix = [[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]]

    return matrix


def rotation_matrix_from_point(xs, ys, zs):
    target = np.array([xs, ys, zs])
    target = target / lg.norm(target)
    right = np.array([-target[1], target[0], 0])
    right = right / lg.norm(right)
    up = np.cross(target, right).tolist()
    return np.transpose(np.vstack((right, up, target)))


def axis_nr(axis):
    nr = 0 if axis == 'x' else 1 if axis == 'y' else 2 if axis == 'z' else -1
    return nr


def plane_params(point, normal):
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = -(a * point[0] + b * point[1] + c * point[2])
    return a, b, c, d


def plane_params_from_matrix(matrix):
    """Converts 4x4 homo matrix to plane params"""
    z = matrix[0:3, 2]
    t = matrix[0:3, 3]
    return plane_params(t, z)


def distance_to_plane(point, plane_params):
    return np.abs(np.sum(np.multiply(point, plane_params))) / np.sqrt(np.sum(np.square(plane_params[:3])))


def on_plane_projection(point, versor, plane_params):
    nom = -np.sum(np.multiply(point, plane_params))
    denom = (np.sum(np.multiply(plane_params, versor)))
    s = nom / denom
    return point + s * np.float64(versor)


def point_on_line(point, versor, dist):
    return point + dist * versor


def viewpoint_plane_intersection(matrix, plane_params, target_axis='x', right_axis='y'):
    target_axis_nr = axis_nr(target_axis)
    right_axis_nr = axis_nr(right_axis)
    v = matrix[:4, target_axis_nr]
    dt = matrix[:4, right_axis_nr]
    dt = dt / np.linalg.norm(dt)
    t = matrix[:4, 3]

    pt_1 = on_plane_projection(t, v, plane_params)
    pt_2 = on_plane_projection(t + dt, v, plane_params)

    versor = (pt_2 - pt_1) / np.linalg.norm(pt_2 - pt_1)
    return versor, pt_1


def log_map(rot):
    theta = np.arccos((np.trace(rot) - 1) / 2)
    if theta < 1e-5:
        coeff = 1.0
    else:
        coeff = theta / np.sin(theta) / 2

    skew_matrix = coeff * (rot - np.transpose(rot))

    return inv_skew_symmetric(skew_matrix)


def exp_map(vector):
    theta = np.sqrt(np.sum(np.square(vector)))
    skew = skew_symmetric(vector)
    if theta < 1e-5:
        map = np.eye(3)
        map = map + (1 + np.square(theta) / 6 + np.power(theta, 4) / 120) * skew
        map = map + (0.5 - np.square(theta) / 24 + np.power(theta, 4) / 720) * np.matmul(skew, skew)
    else:
        map = np.eye(3)
        map = map + np.sin(theta) / theta * skew
        map = map + ((1 - np.cos(theta)) / np.square(theta)) * np.matmul(skew, skew)

    return map


def skew_symmetric(vector):
    a = vector[0]
    b = vector[1]
    c = vector[2]
    matrix = np.array([[0, -c, b],
                       [c, 0, -a],
                       [-b, a, 0]])
    return matrix


def inv_skew_symmetric(matrix):
    a = matrix[2][1]
    b = matrix[0][2]
    c = matrix[1][0]
    return np.array([a, b, c])
