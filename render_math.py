import math
import numpy as np

class Shader:
    @staticmethod
    def step(a, b):
        return 1 if a > b else 0

class Vector2:
    @staticmethod
    def add(a: np.ndarray, b: np.ndarray):
        return np.array([a[0] + b[0], a[1] + b[1]], dtype=float)
    
    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray):
        return np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    
    @staticmethod
    def mul(a: np.ndarray, b):
        if isinstance(b, float) or isinstance(b, int):
            return np.array([a[0] * b, a[1] * b])
        elif isinstance(b, np.ndarray) or isinstance(b, list):
            return np.array([a[0] * b[0], a[1] * b[1]])
        else:
            raise Exception('b was not correct type.')
    
    @staticmethod
    def div(a: np.ndarray, b: float):
        return np.array([a[0] / b, a[1] / b])

    @staticmethod
    def to_Vector3(a: np.ndarray):
        return np.append(a, 1)

class Vector3:
    # TODO vector3 forward, up, right

    @staticmethod
    def negate(a):
        return np.array([-a[0], -a[1], -a[2]], dtype=float)

    @staticmethod
    def dot(a, b):
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])

    @staticmethod
    def cross(a, b):
        return np.array([((a[1] * b[2]) - (a[2] * b[1])), ((a[2] * b[0]) - (a[0] * b[2])), ((a[0] * b[1]) - (a[1] * b[0]))])
    
    # project a onto b
    @staticmethod
    def project(a, b):
        b_mag = Vector3.magnitude(b)
        return Vector3.mul(b, Vector3.dot(a, b) / (b_mag * b_mag))
    
    @staticmethod
    def magnitude(a: np.ndarray):
        return math.sqrt(Vector3.dot(a, a))

    @staticmethod
    def normalize(a: np.ndarray):
        return Vector3.div(a, Vector3.magnitude(a))

    @staticmethod
    def dist(a: np.ndarray, b: np.ndarray):
        return Vector3.magnitude(Vector3.sub(a, b))

    @staticmethod
    def add(a: np.ndarray, b: np.ndarray):
        return np.array([a[0] + b[0], a[1] + b[1], a[2] + b[2]], dtype=float)
    
    @staticmethod
    def sub(a: np.ndarray, b: np.ndarray):
        return np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=float)
    
    @staticmethod
    def mul(a: np.ndarray, b):
        if isinstance(b, float) or isinstance(b, int):
            return np.array([a[0] * b, a[1] * b, a[2] * b])
        elif isinstance(b, np.ndarray) or isinstance(b, list):
            return np.array([a[0] * b[0], a[1] * b[1], a[2] * b[2]])
        else:
            raise Exception('b was not correct type.')
    
    @staticmethod
    def div(a: np.ndarray, b: float):
        return np.array([a[0] / b, a[1] / b, a[2] / b])
    
    @staticmethod
    def equals(a: np.ndarray, b: np.ndarray):
        return abs(a[0] - b[0]) < 0.0001 and abs(a[1] - b[1]) < 0.0001 and abs(a[2] - b[2]) < 0.0001
    
    @staticmethod
    def to_vertical(a: np.ndarray):
        return np.swapaxes([a], 0, 1)
    
    @staticmethod
    def to_horizontal(a: np.ndarray):
        return np.swapaxes(a, 0, 1)[0]

    @staticmethod
    def to_Vector2(a: np.ndarray):
        return a[:2]

    @staticmethod
    def to_Vector4(a: np.ndarray):
        return np.append(a, 1)

class Vector4:
    @staticmethod
    def mul(a: np.ndarray, b):
        if isinstance(b, float):
            return np.array([a[0] * b, a[1] * b, a[2] * b, a[3] * b])
        elif isinstance(b, np.ndarray) or isinstance(b, list):
            return np.array([a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]])
        else:
            raise Exception('b was not correct type.')

    @staticmethod
    def div(a: np.ndarray, b: float):
        return np.array([a[0] / b, a[1] / b, a[2] / b, a[3] / b])

    @staticmethod
    def to_vertical(a: np.ndarray):
        return np.swapaxes([a], 0, 1)
    
    @staticmethod
    def to_horizontal(a: np.ndarray):
        return np.swapaxes(a, 0, 1)[0]

    @staticmethod
    def to_Vector3(a: np.ndarray):
        return a[:3]

class Matrix:
    # Overwrite part of a with b. 
    # Use column and row to choose where to begin replacing.
    @staticmethod
    def overwrite(a: np.ndarray, b: np.ndarray, column: int, row: int):
        new_matrix = a.copy()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                new_matrix[i + column][j + row] = b[i][j]
        return new_matrix

    # inclusive from and to
    @staticmethod
    def submatrix(a, column_from, row_from, column_to, row_to):
        new_matrix = np.ndarray(shape=(1 + column_to - column_from, 1 + row_to - row_from), dtype=float)
        for i in range(column_from, column_to + 1):
            for j in range(row_from, row_to + 1):
                new_matrix[i - column_from][j - row_from] = a[i][j]
        return new_matrix

    @staticmethod
    def negate(a: np.ndarray):
        new_matrix = np.ndarray(shape=a.shape, dtype=float)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                new_matrix[i][j] = -a[i][j]
        return new_matrix