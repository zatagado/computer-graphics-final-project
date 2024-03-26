import numpy as np
import math
from render_math import Vector3, Vector4, Matrix

class Transform:
    """Allows a mesh to manipulated. The model matrix allows for 
    transformations and rotations. Scaling is not implemented.
    """

    def __init__(self):
        """Initializes the transform class.
        """
        self.model_matrix = np.identity(4, dtype=float)

    def transformation_matrix(self):
        """Returns a 4x4 Numpy matrix that represents the transformation 
        matrix.
        """
        return self.model_matrix

    def get_position(self):
        """Gets the position of the transform.
        """
        return np.array([self.model_matrix[0][3], self.model_matrix[1][3], \
            self.model_matrix[2][3]], dtype=float)

    def set_position(self, x, y=None, z=None):
        """This method takes three scalars (x, y, and z) as input, and updates 
        the Transform object's internal state to represent a new position at 
        (x, y, z).
        """
        if isinstance(x, float) or isinstance(x, int):
            self.model_matrix[0][3] = x
            self.model_matrix[1][3] = y
            self.model_matrix[2][3] = z
        else:
            self.model_matrix[0][3] = x[0]
            self.model_matrix[1][3] = x[1]
            self.model_matrix[2][3] = x[2]

    def __rotate_x(self, degrees):
        radians = math.radians(degrees)
        return np.array([[1, 0, 0], [0, math.cos(radians), -math.sin(radians)], [0, math.sin(radians), math.cos(radians)]], dtype=float)
    
    def __rotate_y(self, degrees):
        radians = math.radians(degrees)
        return np.array([[math.cos(radians), 0, math.sin(radians)], [0, 1, 0], [-math.sin(radians), 0, math.cos(radians)]], dtype=float)
    
    def __rotate_z(self, degrees):
        radians = math.radians(degrees)
        return np.array([[math.cos(radians), -math.sin(radians), 0], [math.sin(radians), math.cos(radians), 0], [0, 0, 1]], dtype=float)

    def set_rotation(self, x, y, z):
        """This method takes three scalars (x, y, and z) as input, and updates 
        the Transform object's interal rotation state. The input values x, y, 
        and z are expected to be degrees values between 0.0 and 360.0 (there 
        is no need to check this) , and represent the amount of rotation 
        around the x, y, and z axis respectively. The rotation should be set 
        using the ZYX order of rotation.
        """
        rotation_matrix = np.matmul(np.matmul(self.__rotate_x(x), self.__rotate_y(y)), self.__rotate_z(z))
        self.model_matrix = Matrix.overwrite(self.model_matrix, rotation_matrix, 0, 0)

    def inverse_matrix(self):
        """This method returns a 4x4 Numpy matrix that is the inverse of the 
        transformation matrix.
        """
        # transpose of rotation matrix is inverse
        inverse_rotation = Matrix.overwrite(np.identity(4, dtype=float), np.transpose(\
            Matrix.submatrix(self.model_matrix, 0, 0, 2, 2)), 0, 0)
        # negative translation vector is inverse
        inverse_translation = Matrix.overwrite(np.identity(4, dtype=float), Matrix.negate(\
            Matrix.submatrix(self.model_matrix, 0, 3, 2, 3)), 0, 3)
        
        return np.matmul(inverse_rotation, inverse_translation)
    
    def apply_to_point(self, p: np.ndarray):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in space as input. It then applies the transformation matrix to 
        it, and returns the resulting 3 element Numpy array.
        """
        p_4D = Vector4.to_vertical(Vector3.to_Vector4(p))
        result_4D = np.matmul(self.model_matrix, p_4D)
        return Vector4.to_Vector3(Vector4.to_horizontal(result_4D))

    def apply_inverse_to_point(self, p: np.ndarray):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in space as input. It then applies the inverse transformation 
        matrix to it, and returns the resulting 3 element Numpy array.
        """
        p_4D = Vector4.to_vertical(Vector3.to_Vector4(p))
        result_4D = np.matmul(self.inverse_matrix(), p_4D)
        return Vector4.to_Vector3(Vector4.to_horizontal(result_4D))

    def apply_to_normal(self, n: np.ndarray):
        """This method takes a 3 element Numpy array, n, that represents a 3D 
        normal vector. It then applies the transform's rotation to it, and 
        returns the resulting 3 element Numpy array. The resulting array 
        should be normalized and should not be affected by any positional 
        component within the transform.
        """
        rotation = Matrix.submatrix(self.model_matrix, 0, 0, 2, 2)
        normal = np.matmul(rotation, Vector3.to_vertical(n))
        return Vector3.normalize(Vector3.to_horizontal(normal))
    
    # Uses Rodrigues' rotation formula to rotate a number of degrees around a vector axis and applied to the model matrix
    def set_axis_rotation(self, axis: np.ndarray, rotation):
        """This method takes a 3D numpy array as the axis input, and a scalar 
        in degrees as the rotation input. This method calculates the rotation 
        resulting from the amount of input rotation rotating around the input 
        axis and update its internal rotation representation as such.
        """
        axis = Vector3.normalize(axis)
        k = np.zeros((3, 3))
        k[0][1] = -axis[2]
        k[0][2] = axis[1]
        k[1][0] = axis[2]
        k[1][2] = -axis[0]
        k[2][0] = -axis[1]
        k[2][1] = axis[0]
        radians = math.radians(rotation)
        self.model_matrix = Matrix.overwrite(self.model_matrix, np.add(np.add(np.identity(3, dtype=float)\
            , k * math.sin(radians)), np.matmul(k, k) * (1 - math.cos(radians))), 0, 0)
        
    # modified from source: https://stackoverflow.com/questions/18558910/direction-vector-to-rotation-matrix
    def set_rotation_towards(self, direction):
        direction = Vector3.normalize(Vector3.negate(direction)) # TODO not sure why this is needed
        up=[0, 0, 1]
        xAxis = None
        zAxis = None
        if direction[2] == 1:
            xAxis = [1, 0, 0]
            zAxis = [0, -1, 0]
        elif direction[2] == -1:
            xAxis = [1, 0, 0]
            zAxis = [0, 1, 0]
        else:
            xAxis = Vector3.normalize(Vector3.cross(direction, up))
            zAxis = Vector3.normalize(Vector3.cross(xAxis, direction))

        self.model_matrix[0][0] = xAxis[0]
        self.model_matrix[1][0] = xAxis[1]
        self.model_matrix[2][0] = xAxis[2]

        self.model_matrix[0][1] = direction[0]
        self.model_matrix[1][1] = direction[1]
        self.model_matrix[2][1] = direction[2]

        self.model_matrix[0][2] = zAxis[0]
        self.model_matrix[1][2] = zAxis[1]
        self.model_matrix[2][2] = zAxis[2]