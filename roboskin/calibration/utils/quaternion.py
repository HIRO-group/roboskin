import math
import numpy as np
import pyquaternion as pyqt
from geometry_msgs.msg import Quaternion


def tf_to_pyqt(q):
    """
    Convert a tf quaternion to a pyqt array.
    Order of a tf quaternion
    q = (x, y, z, w)

    Order of a pyqt quaternion
    q = (w, x, y, z)
    """
    return pyqt.Quaternion(w=q.w, x=q.x, y=q.y, z=q.z)


def pyqt_to_tf(q):
    """
    Convert a pyqt quaternion to a tf array.
    Order of a tf quaternion
    q = (x, y, z, w)

    Order of a pyqt quaternion
    q = (w, x, y, z)
    """
    q = q.elements
    return Quaternion(q[1], q[2], q[3], q[0])


def pyqt_to_np(q):
    """
    Convert a pyqt quaternion to a numpy array.
    Order of a numpy quaternion
    q = (x, y, z, w)

    Order of a pyqt quaternion
    q = (w, x, y, z)
    """
    q = q.elements
    return np.array([q[1], q[2], q[3], q[0]])


def np_to_pyqt(q):
    """
    Convert a numpyp quaternion to a pyqt array.
    Order of a numpy quaternion
    q = (x, y, z, w)

    Order of a pyqt quaternion
    q = (w, x, y, z)
    """
    return pyqt.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def quaternion_l2_distance(q1, q2):
    """
    A metric for computing the distance
    between 2 quaternions.
    sources:
    - https://fgiesen.wordpress.com/2013/01/07/small-note-on-quaternion-distance-metrics/
    - http://kieranwynn.github.io/pyquaternion/#accessing-individual-elements
    """
    return 2*(1 - np.dot(q1.elements, q2.elements))


def quaternion_from_two_vectors(source, target):
    """
    Computes the quaternion from vector `source`
    to vector `target`.

    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)

    axis = np.cross(source, target)
    costh = np.dot(source, target)

    angle = np.arccos(costh)

    if angle == 0.0:
        return pyqt.Quaternion()

    return pyqt.Quaternion(axis=axis, angle=angle)


def angle_between_quaternions(q_1: np.ndarray, q_2: np.ndarray, output_in_degrees: bool = False) -> float:  # noqa: E999
    r"""
    Angle between quaternions a and b in degrees. Please note the input quaternions should be of
    form np.ndarray([x, y, z, w]).
    The formula for angle between quaternions is:

    .. math::
        \theta = \cos^{-1}\bigl(2\langle q_1,q_2\rangle^2 -1\bigr)

    where ⟨q1,q2⟩ denotes the inner product of the corresponding quaternions:

    .. math::
        \langle a_1 +b_1 \textbf{i} + c_1 \textbf{j} + d_1 \textbf{k}, \\
        a_2 + b_2 \textbf{i} + c_2 \textbf{j} + d_2 \textbf{k}\rangle \\
        = a_1a_2 + b_1b_2 + c_1 c_2 + d_1d_2.

    Reference: https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q_1: np.ndarray
        Quaternion a
    :param q_2: np.ndarray
        Quaternion b
    :return: float
        Angle between quaternions in degrees
    """
    if not (math.isclose(np.linalg.norm(q_1), 1.0, abs_tol=0.1) and math.isclose(np.linalg.norm(q_2), 1.0, abs_tol=0.1)):
        raise Exception("Please only pass unit quaternions")
    angle = np.arccos(2 * ((q_1 @ q_2) ** 2) - 1)  # noqa: E999
    if np.isnan(angle):
        # usually some values overflow 1. arc-cos isn't defined in range > 1
        # So it's just an overflow error and the angle can be safely be assumed zero
        return 0.0
    if output_in_degrees:
        angle_in_degrees = (angle / np.pi) * 180
        return angle_in_degrees
    return angle
