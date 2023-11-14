from transform import Transform
from render_math import Vector3, Vector4
import numpy as np

class PerspectiveCamera:
    def __init__(self, left, right, bottom, top, near, far):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.near = near
        self.far = far

        # Math for the perspective transform matrix so we don't need to calculate it each time
        self.persp_transform = np.identity(4, dtype=float)
        self.persp_transform[0] = np.array([near, 0, 0, 0]) # TODO change this to positive after class
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
        return (self.left - self.right) / (self.top - self.bottom)

    def project_point(self, p):
        p_view = Vector4.to_vertical(Vector3.to_Vector4(self.transform.apply_inverse_to_point(p)))
        p_persp = np.matmul(self.persp_transform, p_view)
        p_persp = Vector4.div(p_persp, p_persp[3]) #* dividing by the depth value
        p_final = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.ortho_transform, p_persp)))
        # flip the y and z value
        p_final[1], p_final[2] = p_final[2], p_final[1]
        return p_final

    def project_inverse_point(self, p):
        # p_copy = np.array([p[0], p[1], p[2]], dtype=float)
        # p_copy = np.array(p, dtype=float)
        # flip the y and z value
        # p_copy[1], p_copy[2] = p_copy[2], p_copy[1]
        p = np.array([p[0], p[2], p[1]], dtype=float)
        # multiply by inverse orthographic transform
        p_persp = np.matmul(self.inverse_ortho_transform, Vector3.to_vertical(Vector3.to_Vector4(p_copy)))
        
        y = (self.far * self.near) / ((self.near + self.far) - p_persp[1])
        p_persp = Vector4.mul(p_persp, y)
        p_view = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.inverse_persp_transform, p_persp)))
        # multiply by camera transform
        p_world = self.transform.apply_to_point(p_view)
        return p_world

class OrthoCamera:
    # Left is positive because we are looking down -y
    def __init__(self, left, right, bottom, top, near, far):
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
        return (self.left - self.right) / (self.top - self.bottom)

    def project_point(self, p):
        p_view = Vector4.to_vertical(Vector3.to_Vector4(self.transform.apply_inverse_to_point(p)))
        p_ortho = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.ortho_transform, p_view)))
        # flip the y and z value
        p_ortho[1], p_ortho[2] = p_ortho[2], p_ortho[1]
        return p_ortho

    def inverse_project_point(self, p):
        # flip the y and z value
        # p[1], p[2] = p[2], p[1]
        p = np.array([p[0], p[2], p[1]], dtype=float)
        # multiply by inverse orthographic transform
        p_view = Vector4.to_Vector3(Vector4.to_horizontal(np.matmul(self.inverse_ortho_transform, \
            Vector4.to_vertical(Vector3.to_Vector4(p)))))
        # multiply by camera transform
        p_world = self.transform.apply_to_point(p_view)
        return p_world