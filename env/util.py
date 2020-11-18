import numpy as np
import math as math


def angle_of(vector: np.array) -> float:
    return np.arctan2(vector[1], vector[0])


def length_of(vector: np.array) -> float:
    return np.sqrt(vector[0] ** 2 + vector[1] ** 2)


def normalize_vector(vector: np.array) -> np.array:
    length = length_of(vector)
    if not length > 0.0:
        return vector
    else:
        return vector / length


def add_angles(a: float, b: float) -> float:
    return (a + b + 3 * np.pi) % (2.0 * np.pi) - np.pi


def normalize_angle(angle: float) -> float:
    angle = math.fmod(angle + np.pi, 2.0 * np.pi)
    if angle < 0:
        angle += 2.0 * np.pi
    return angle - np.pi


def cartesian_to_polar(x: float, y: float) -> (float, float):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def polar_to_cartesian(rho: float, phi: float) -> (float, float):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def vector_in_torus_space(
        vector_1: np.array,
        vector_2: np.array,
        env_width: float,
        env_height: float) -> (float, float):
    """https://blog.demofox.org/2017/10/01/"""
    dx = vector_2[0] - vector_1[0]
    dy = vector_2[1] - vector_1[1]

    if dx > 0.5 * env_width:
        dx = dx - env_width
    elif dx < -0.5 * env_width:
        dx = env_width + dx

    if dy > 0.5 * env_height:
        dy = dy - env_height
    elif dy < -0.5 * env_height:
        dy = env_height + dy

    return dx, dy


# TODO This is unused right now. Also, weird naming.
def vector_in_torus_space_upgraded(
        vector_1: np.array,
        vector_2: np.array,
        env_width: float,
        env_height: float) -> (float, float):
    """Returns the distance between two vectors in Torus space.
    This function is equal to vector_in_torus_space(above).
    Upgrade: This function allows single and stacked vectors as input.
    """
    distance = (vector_2 - vector_1).T
    dx = distance[0]
    dy = distance[1]

    dx = np.where(abs(dx) > env_width * 0.5, dx - (np.sign(dx) * env_width), dx)
    dy = np.where(abs(dy) > env_width * 0.5, dy - (np.sign(dy) * env_height), dy)

    return dx, dy


def shortest_angle_between(source_angle: float, target_angle: float) -> float:
    """Returns the shortest way to turn between two angles
    Source: https://stackoverflow.com/questions/1878907/
    """
    return (target_angle - source_angle + np.pi) % (2 * np.pi) - np.pi


def scale(value, input_min, input_max, output_min, output_max):
    """https://stackoverflow.com/questions/929103/"""
    input_range = input_max - input_min
    output_range = output_max - output_min
    return (((value - input_min) * output_range) / input_range) + output_min
