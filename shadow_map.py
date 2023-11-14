import numpy as np
import math
from light import DirectionalLight
from camera import OrthoCamera
from render_math import Vector3

class ShadowMap:
    def __init__(self, light: DirectionalLight, meshes, orthoCamera: OrthoCamera): 
        self.orthoCamera = orthoCamera
        self.orthoCamera.transform.set_rotation_towards(light.transform.apply_to_normal(Vector3.negate(Vector3.forward())))
        self.depth_buffer = np.full((0, 0), -math.inf, dtype=float) #? what should the width and height be 
        # TODO scale this to cover the entire screen
        # values from 0 - 1
        pass

    def checkOcclusion(self, p): #* point must be within world space
        # TODO point is some distance away from the eye
        # TODO can find out xyz in world space of pixel within screen coordinates
        # TODO can figure out the pixel that xyz corresponds to within the shadow map
        # TODO then find out if the depth within the depth map is the same as the depth of that pixel that was just checked
        
        # TODO use the orthographic projection for straight rays

        # TODO add bias to occluder estimation
        pass