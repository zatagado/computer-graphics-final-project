import numpy as np
import math
from light import DirectionalLight
from camera import OrthoCamera
from render_math import Vector3
from mesh import Mesh

from debug import Store

class ShadowMap:
    def device_to_screen(self, p):
        p_screen = Vector3.to_Vector2(p)
        p_screen[0] = (p_screen[0] + 1) * (self.resolution[0] / 2)
        p_screen[1] = (p_screen[1] + 1) * (self.resolution[1] / 2)
        return p_screen
    
    def __init__(self, meshes, light: DirectionalLight, orthoCamera: OrthoCamera, resolution, bias): 
        def get_pixel_bounds(screen_coords_verts, width, height):
            bounding_rect_min = [screen_coords_verts[0][0], screen_coords_verts[0][1]]
            bounding_rect_max = [screen_coords_verts[0][0], screen_coords_verts[0][1]]
            for vert in screen_coords_verts:
                if vert[0] < bounding_rect_min[0]:
                    bounding_rect_min[0] = vert[0]
                if vert[0] > bounding_rect_max[0]:
                    bounding_rect_max[0] = vert[0]
                if vert[1] < bounding_rect_min[1]:
                    bounding_rect_min[1] = vert[1]
                if vert[1] > bounding_rect_max[1]:
                    bounding_rect_max[1] = vert[1]

            if int(bounding_rect_min[0]) >= 0:
                if int(bounding_rect_min[0]) < width:
                    bounding_rect_min[0] = int(bounding_rect_min[0])
                else:
                    bounding_rect_min[0] = width
            else:
                bounding_rect_min[0] = 0
            
            if int(bounding_rect_max[0]) >= 0:
                if int(bounding_rect_max[0]) < width:
                    bounding_rect_max[0] = int(bounding_rect_max[0]) + 1
                else:
                    bounding_rect_max[0] = width
            else:
                bounding_rect_max[0] = 0

            if int(bounding_rect_min[1]) >= 0:
                if int(bounding_rect_min[1]) < height:
                    bounding_rect_min[1] = int(bounding_rect_min[1])
                else:
                    bounding_rect_min[1] = height
            else:
                bounding_rect_min[1] = 0
            
            if int(bounding_rect_max[1]) >= 0:
                if int(bounding_rect_max[1]) < height:
                    bounding_rect_max[1] = int(bounding_rect_max[1]) + 1
                else:
                    bounding_rect_max[1] = height
            else:
                bounding_rect_max[1] = 0

            return [bounding_rect_min, bounding_rect_max]

        def fill_depth_buffer(meshes, orthoCamera, resolution):
            depth_buffer = np.full(resolution, -math.inf, dtype=float)

            for i in range(len(meshes)):
                mesh: Mesh = meshes[i]

                world_verts = [mesh.transform.apply_to_point(p) for p in mesh.verts]
                ndc_verts = [orthoCamera.project_point(p) for p in world_verts]
                screen_verts = [self.device_to_screen(p) for p in ndc_verts]

                for j in range(len(mesh.faces)):
                    face = mesh.faces[j]
                    ndc_tri = [ndc_verts[face[0]], ndc_verts[face[1]], ndc_verts[face[2]]]
                    screen_tri = [screen_verts[face[0]], screen_verts[face[1]], screen_verts[face[2]]]

                    # cull triangle if normal is facing away from camera
                    cross = Vector3.cross(Vector3.normalize(Vector3.sub(ndc_tri[1], ndc_tri[0])), \
                        Vector3.normalize(Vector3.sub(ndc_tri[2], ndc_tri[0])))
                    if cross[2] < 0: # TODO maybe eventually flip the normals
                        continue

                    # get pixel bounds for which areas of pixels should be drawn
                    bounds = get_pixel_bounds(screen_tri, resolution[0], resolution[1])
                    x_min = bounds[0][0]
                    x_max = bounds[1][0]
                    y_min = bounds[0][1]
                    y_max = bounds[1][1]

                    x_a = screen_tri[0][0]
                    x_b = screen_tri[1][0]
                    x_c = screen_tri[2][0]
                    y_a = screen_tri[0][1]
                    y_b = screen_tri[1][1]
                    y_c = screen_tri[2][1]

                    for x in range(x_min, x_max):
                        for y in range(y_min, y_max):
                            # barycentric coordinate checking
                            # ? What happens when there is a negative divisor
                            gammaDivisor = (((y_a - y_b) * x_c) + ((x_b - x_a) * y_c) + (x_a * y_b) - (x_b * y_a))
                            if gammaDivisor == 0:
                                gammaDivisor = 0.0000001
                            gamma = (((y_a - y_b) * x) + ((x_b - x_a) * y) + (x_a * y_b) - (x_b * y_a)) / gammaDivisor
                                
                            betaDivisor = (((y_a - y_c) * x_b) + ((x_c - x_a) * y_b) + (x_a * y_c) - (x_c * y_a))
                            if betaDivisor == 0:
                                betaDivisor = 0.0000001
                            beta = (((y_a - y_c) * x) + ((x_c - x_a) * y) + (x_a * y_c) - (x_c * y_a)) / betaDivisor

                            alpha = 1 - beta - gamma

                            # ignore pixel outside of triangle
                            if gamma < 0 or beta < 0 or alpha < 0:
                                continue

                            # ignore pixel outside near far bounds
                            depth = Vector3.add(Vector3.mul(ndc_tri[0], alpha), Vector3.add(Vector3.mul(ndc_tri[1], beta), Vector3.mul(ndc_tri[2], gamma)))[2]
                            if depth > 1 or depth < -1: #? should the first be 1 instead of 0????
                                continue

                            # render the pixel depending on the depth of the pixel
                            if (depth + 1) / 2 <= depth_buffer[x, y]:
                                continue
                            
                            # add depth of pixel
                            depth_buffer[x, y] = (depth + 1) / 2
            
            return depth_buffer
            
        #* We must scale the orthoCamera to cover the entire screen. The shadow mapper does not currently do this automatically.
        #* We don't care about the position of the camera. Instead move the camera near plane.
        self.orthoCamera = orthoCamera
        # Rotate the camera the same direction the light is facing
        self.orthoCamera.transform.set_rotation_towards(light.transform.apply_to_normal(Vector3.negate(Vector3.forward())))
        self.orthoCamera.transform.set_position(light.transform.get_position()) # ! The position probably doesn't matter
        self.resolution = resolution
        self.bias = bias
        # Fill the depth buffer with depths from the viewpoint of the light source
        self.depth_buffer = fill_depth_buffer(meshes, orthoCamera, resolution)

    def check_occlusion(self, p): #* point must be within world space
        # TODO point is some distance away from the eye
        # TODO can find out xyz in world space of pixel within screen coordinates
        # TODO can figure out the pixel that xyz corresponds to within the shadow map
        # TODO then find out if the depth within the depth map is the same as the depth of that pixel that was just checked
        
        # TODO use the orthographic projection for straight rays

        # TODO add bias to occluder estimation

        # TODO return 0 if dark and 1 if light

        ndc_vert = self.orthoCamera.project_point(p) #! this outputs the wrong position
        screen_vert = self.device_to_screen(ndc_vert)

        # ? where should the pixel lie, floor or ceil or somewhere in the middle?
        x = math.floor(screen_vert[0])
        y = math.floor(screen_vert[1]) #? is there some issue with the pixel being flipped
        depth = ndc_vert[2]
        if depth > 1 or depth < -1: 
            return 1
        if Store.boolean:
            print(ndc_vert)
            print(screen_vert)
            print(f'x: {x}\ny: {y}')
            Store.boolean = False
        return 0 if ((depth + 1) / 2) + self.bias < self.depth_buffer[x, y] else 1 # TODO bias 