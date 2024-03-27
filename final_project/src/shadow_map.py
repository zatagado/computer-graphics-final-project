import numpy as np
import math
from light import DirectionalLight
from camera import OrthoCamera
from render_math import Vector3
from mesh import Mesh

class ShadowMap:
    """This class is used to generate casted shadows upon the surfaces of 
    meshes. It does this by using an orthographic camera facing in the 
    direction of the light, then generating a depth map. The main camera 
    compares the shadow map transformed depth of pixels it renders to the 
    shadow depth map. ShadowMaps are currently only implemented to use 
    directional lights and will not work with point lights.
    """

    def device_to_screen(self, p):
        """Converts a point from device coordinate space to screen coordinate 
        space.
        """
        p_screen = Vector3.to_Vector2(p)
        p_screen[0] = (p_screen[0] + 1) * (self.resolution[0] / 2)
        p_screen[1] = (p_screen[1] + 1) * (self.resolution[1] / 2)
        return p_screen
    
    def __init__(self, meshes, light: DirectionalLight, orthoCamera: OrthoCamera, resolution, bias): 
        """Initializes the ShadowMap object and generates the shadow map. 
        """
        def get_pixel_bounds(screen_coords_verts, width, height):
            """Gets the x and y pixel bounds of the mesh from its bounding box.
            Used for optimization so we know we do not have to modify pixels 
            outside of those bounds for the current mesh.
            """
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
            """Actually generates the depth buffer for the shadow map from the 
            perspective of the orthographic camera. Takes a resolution for the 
            depth buffer./
            """
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
                    
                    # if cross[2] < 0: #! Regular normals
                    #     continue
                    if cross[2] >= 0: #! Flipped normals
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
                            if depth > 1 or depth < -1:
                                continue

                            # render the pixel depending on the depth of the pixel
                            if (depth + 1) / 2 <= depth_buffer[x, y]:
                                continue
                            
                            # add depth of pixel
                            depth_buffer[x, y] = (depth + 1) / 2
            
            return depth_buffer
        
        if isinstance(light, DirectionalLight):
            #* We must scale the orthoCamera to cover the entire screen. The shadow mapper does not currently do this automatically.
            #* We don't care about the position of the camera. Instead move the camera near plane.
            self.orthoCamera = orthoCamera
            # Rotate the camera the same direction the light is facing
            self.orthoCamera.transform.set_rotation_towards(light.transform.apply_to_normal(Vector3.negate(Vector3.forward())))

            self.resolution = resolution
            self.bias = bias
            # Fill the depth buffer with depths from the viewpoint of the light source
            self.depth_buffer = fill_depth_buffer(meshes, orthoCamera, resolution)

    def check_occlusion(self, p): #* point must be within world space
        """Checks the shadow map to see if the given world space point is
        occluded by another surface. Returns 0 if point is occluded, 1 
        otherwise. 
        """
        ndc_vert = self.orthoCamera.project_point(p) 
        
        if ndc_vert[0] < -1 or 1 < ndc_vert[0] or ndc_vert[1] < -1 or 1 < ndc_vert[1]:
            return 1

        depth = ndc_vert[2]
        if depth > 1 or depth < -1: 
            return 1

        screen_vert = self.device_to_screen(ndc_vert)

        x = math.floor(screen_vert[0])
        y = math.floor(screen_vert[1])

        return 0 if ((depth + 1) / 2) + self.bias < self.depth_buffer[x, y] else 1