import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from mpl_toolkits.mplot3d import Axes3D


def py_ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang)


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def plot_line(ax, pt_1, pt_2, color):
    xs = [pt_1[0], pt_2[0]]
    ys = [pt_1[1], pt_2[1]]
    zs = [pt_1[2], pt_2[2]]
    ax.plot3D(xs, ys, zs, color)


def plot_transforms(tfs, axis_length=1.0, ax=None):
    if (ax == None):
        ax = plt.axes(projection='3d')
    for tf in tfs:
        x = tf[0][3]
        y = tf[1][3]
        z = tf[2][3]
        eye = np.array([x, y, z])
        tf = tf.transpose()
        x_axis = tf[0][:3]
        y_axis = tf[1][:3]
        z_axis = tf[2][:3]
        x_end = np.add(eye, axis_length * x_axis)
        y_end = np.add(eye, axis_length * y_axis)
        z_end = np.add(eye, axis_length * z_axis)
        plot_line(ax, eye, x_end, 'Red')
        plot_line(ax, eye, y_end, 'Green')
        plot_line(ax, eye, z_end, 'Blue')


def plot_points(xs, ys, zs, ax=None):
    if (ax == None):
        ax = plt.axes(projection='3d')
    ax.scatter(xs, ys, zs, s=50, alpha=1.0)


def plot(transforms=None, points=None, origin=None, radius=None):
    ax = plt.axes(projection='3d')
    if transforms is not None:
        plot_transforms(transforms, 0.3, ax)

    if points is not None:
        plot_points(points[0], points[1], points[2], ax)

    ax.set_aspect('equal')
    if origin is None or radius is None:
        set_axes_equal(ax)
    else:
        assert len(origin) == 3
        set_axes_radius(ax, origin, radius)
