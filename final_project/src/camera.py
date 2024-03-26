from transform import Transform
from render_math import Vector3, Vector4
import numpy as np

class PerspectiveCamera:

    def __init__(self, left, right, bottom, top, near, far):
        """The constructor takes six floats as arguments: left , right, 
        bottom, top, near, and far. These arguments define the orthographic 
        projection of the camera used to construct the orthographic 
        transformation. You can then use the near and far values to construct 
        the perspective matrix. Using these two matrices is how you then 
        convert a point from camera space to device space in the method 
        project_point. You should also construct the inverse matrices for both 
        the orthographic and projective transformations, as those will both be 
        used in the method project_inverse_point. The camera transform should 
        be initialized with the Transform default constructor.
        """
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

        # Math for the perspective transform matrix so we don't need to calculate it each time
        self.persp_transform = np.identity(4, dtype=float)
        self.persp_transform[0] = np.array([near, 0, 0, 0])
        self.persp_transform[1] = np.array([0, near + far, 0, -(far * near)])
        self.persp_transform[2] = np.array([0, 0, near, 0])
        self.persp_transform[3] = np.array([0, 1, 0, 0])
        
        # Math for the orthographic transform matrix so we don't need to calculate it each time
        self.ortho_transform = np.identity(4, dtype=float)
        self.ortho_transform[0] = np.array([2 / (right - left), 0, 0, -((right + left) / (right - left))])
        self.ortho_transform[1] = np.array([0, 2 / (near - far), 0, -((near + far) / (near - far))])
        self.ortho_transform[2] = np.array([0, 0, 2 / (top - bottom), -((top + bottom) / (top - bottom))])

        # Math for the inverse perspective transform matrix
        self.inverse_persp_transform = np.linalg.inv(self.persp_transform)
        
        # Math for the inverse orthographic transform matrix
        self.inverse_ortho_transform = np.linalg.inv(self.ortho_transform)

        self.transform = Transform()

    def ratio(self):
        """This method simply returns a float that is the ratio of the camera 
        projection plane's width to height.
        """
        return (self.left - self.right) / (self.top - self.bottom)

    def project_point(self, p):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in world space as input. It then transforms p to the camera 
        coordinate system before performing the perspective projection using 
        and returns the resulting 3 element Numpy array.
        """
        p_view = Vector4.to_vertical(Vector3.to_Vector4(self.transform.apply_inverse_to_point(p)))
        p_persp = np.matmul(self.persp_transform, p_view)
        p_persp = Vector4.div(p_persp, p_persp[3]) #* dividing by the depth value
        p_final = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.ortho_transform, p_persp)))
        # flip the y and z value
        p_final[1], p_final[2] = p_final[2], p_final[1]
        return p_final

    def inverse_project_point(self, p):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in normalized device coordinates as input. It then transforms p 
        to the camera coordinate system before transforming back to world 
        space returns the resulting 3 element Numpy array.
        """
        p = np.array([p[0], p[2], p[1]], dtype=float)
        # multiply by inverse orthographic transform
        p_persp = np.matmul(self.inverse_ortho_transform, Vector3.to_vertical(Vector3.to_Vector4(p)))
        
        y = (self.far * self.near) / ((self.near + self.far) - p_persp[1])
        p_persp = Vector4.mul(p_persp, y)
        p_view = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.inverse_persp_transform, p_persp)))
        # multiply by camera transform
        p_world = self.transform.apply_to_point(p_view)
        return p_world

class OrthoCamera:
    # Left is positive because we are looking down -y
    def __init__(self, left, right, bottom, top, near, far):
        """The constructor takes six floats as arguments: left , right, 
        bottom, top, near, and far. These arguments define the orthographic 
        projection of the camera used to construct ortho_transform. The camera 
        transform is initialized with the Transform default constructor.
        """
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far
        
        # Math for the orthographic transform matrix so we don't need to calculate it each time
        self.ortho_transform = np.identity(4, dtype=float)
        self.ortho_transform[0] = np.array([2 / (right - left), 0, 0, -((right + left) / (right - left))])
        self.ortho_transform[1] = np.array([0, 2 / (near - far), 0, -((near + far) / (near - far))])
        self.ortho_transform[2] = np.array([0, 0, 2 / (top - bottom), -((top + bottom) / (top - bottom))])

        # Math for the inverse orthographic transform matrix
        self.inverse_ortho_transform = np.linalg.inv(self.ortho_transform)

        self.transform = Transform()

    def ratio(self):
        """This method simply returns a float that is the ratio of the camera 
        projection plane's width to height.
        """
        return (self.left - self.right) / (self.top - self.bottom)

    def project_point(self, p):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in world space as input. It then transforms p to the camera 
        coordinate system before performing an orthographic projection using 
        ortho_transform and returns the resulting 3 element Numpy array.
        """
        p_view = Vector4.to_vertical(Vector3.to_Vector4(self.transform.apply_inverse_to_point(p)))
        p_ortho = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.ortho_transform, p_view)))
        # flip the y and z value
        p_ortho[1], p_ortho[2] = p_ortho[2], p_ortho[1]
        return p_ortho

    def inverse_project_point(self, p):
        """This method takes a 3 element Numpy array, p, that represents a 3D 
        point in normalized device coordinates as input. It then transforms p 
        to the camera coordinate system before transforming back to world 
        space using inverse_ortho_transform and returns the resulting 3 
        element Numpy array.
        """
        # flip the y and z value
        # p[1], p[2] = p[2], p[1]
        p = np.array([p[0], p[2], p[1]], dtype=float)
        # multiply by inverse orthographic transform
        p_view = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.inverse_ortho_transform, \
            Vector4.to_vertical(Vector3.to_Vector4(p)))))
        # multiply by camera transform
        p_world = self.transform.apply_to_point(p_view)
        return p_world