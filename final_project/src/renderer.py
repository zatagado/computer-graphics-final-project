from screen import Screen
from mesh import Mesh
from render_math import Vector2, Vector3, Vector4, Shader
from light import PointLight, DirectionalLight
from camera import PerspectiveCamera
import numpy as np
import math
import pattern

def quantize_color(target: np.ndarray):
    """Quantize a color to a more limited palette
    """
    bucket_count = 5
    return 255 / (bucket_count - 1) * (target * bucket_count // 256)

def skew_color(target: np.ndarray, amount):
    """Not sure what to call this, but it distorts colors in an interesting 
    way. The brighter a color, the more it will be distorted.
    """
    s = (target[0] + target[1] + target[2]) / (255 * 3) * amount
    return np.array([target[2] * s + target[0] * (1 - s), target[0] * s + target[1] * (1 - s), target[1] * s + target[2] * (1 - s)])

class Renderer:
    def __init__(self, screen: Screen, camera, light, shadow_map=None):
        """The class constructor takes a screen object (of type Screen), and 
        camera object (either of type OrthoCamera or PerspectiveCamera), and 
        a mesh object (of type Mesh) and stores them.
        """
        self.screen = screen
        self.camera = camera
        self.light = light
        self.shadow_map = shadow_map

    def render(self, passes, bg_color, ambient_light):
        """Executes the basic render loop to construct an image buffer. It 
        will then draw that image buffer to the screen object using the 
        screen.draw method, but it will not run the pygame loop (the calling 
        function will call screen.show). Takes list of passes that define the 
        shading type.
        """
        def get_pixel_bounds(screen_coords_verts, screen_width, screen_height):
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
                if int(bounding_rect_min[0]) < screen_width:
                    bounding_rect_min[0] = int(bounding_rect_min[0])
                else:
                    bounding_rect_min[0] = screen_width
            else:
                bounding_rect_min[0] = 0
            
            if int(bounding_rect_max[0]) >= 0:
                if int(bounding_rect_max[0]) < screen_width:
                    bounding_rect_max[0] = int(bounding_rect_max[0]) + 1
                else:
                    bounding_rect_max[0] = screen_width
            else:
                bounding_rect_max[0] = 0

            if int(bounding_rect_min[1]) >= 0:
                if int(bounding_rect_min[1]) < screen_height:
                    bounding_rect_min[1] = int(bounding_rect_min[1])
                else:
                    bounding_rect_min[1] = screen_height
            else:
                bounding_rect_min[1] = 0
            
            if int(bounding_rect_max[1]) >= 0:
                if int(bounding_rect_max[1]) < screen_height:
                    bounding_rect_max[1] = int(bounding_rect_max[1]) + 1
                else:
                    bounding_rect_max[1] = screen_height
            else:
                bounding_rect_max[1] = 0

            return [bounding_rect_min, bounding_rect_max]

        def perspective_correction_w(camera, p):
            """Applies perspective correction for perspective correct textures 
            and shadows. Without this correction, textures on the mesh will not 
            have a vanishing point due to perspective.
            """
            p_copy = np.array(p, dtype=float)
            p_copy[1], p_copy[2] = p_copy[2], p_copy[1]
            p_persp = np.matmul(camera.inverse_ortho_transform, Vector4.to_vertical(Vector3.to_Vector4(p_copy)))
            w = (camera.far * camera.near) / ((camera.near + camera.far) - p_persp[1][0])
            return w

        def shade_flat(light, ambient_light, mesh, world_tri, normal, alpha, beta, gamma):
            """Applies flat shading to the mesh using its color properties. 
            Allows for shadows on mesh but no support for shadow maps. No 
            support for textures.
            """
            p = Vector3.add(Vector3.mul(world_tri[0], alpha), \
                Vector3.add(Vector3.mul(world_tri[1], beta), Vector3.mul(world_tri[2], gamma)))
            
            n = mesh.transform.apply_to_normal(normal)
            l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

            o = mesh.diffuse_color
            kd = mesh.kd
            phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi), None, 1)

            l_color = light.color
            if isinstance(light, DirectionalLight):
                id = l_color
                d = Vector3.mul(id, phi_d)
                a = Vector3.mul(ambient_light, mesh.ka)

                return Vector3.clamp(Vector3.mul(Vector3.add(a, d), 255), None, 255)
            elif isinstance(light, PointLight):
                l_distance = Vector3.dist(light.transform.get_position(), p)
                l_intensity = light.intensity
                id = Vector3.div(Vector3.mul(l_color, l_intensity), (l_distance * l_distance))
                d = Vector3.mul(id, phi_d)
                a = Vector3.mul(ambient_light, mesh.ka)

                return Vector3.clamp(Vector3.mul(Vector3.add(a, d), 255), None, 255)

        def shade_barycentric(alpha, beta, gamma):
            """Applies barycentric shading to the mesh. The first vertex is 
            red, the second green, and the third blue. All pixel colors on the 
            triangle are interpolated between the colors of its vertices 
            according to the barycentric coordinate system. 
            """
            return (255 * alpha, 255 * beta, 255 * gamma)
        
        def shade_depth(depth):
            """Applies depth shading to the mesh. Pixel colors are grayscale 
            with pixels close the near plane being white and pixels close to 
            the far plane being black with pixels in between being 
            interpolated.
            """
            depth_norm = (depth + 1) / 2
            return (255 * depth_norm, 255 * depth_norm, 255 * depth_norm)
        
        def shade_phong_blinn(camera, light, ambient_light, shadow_map, mesh, ndc_tri, world_tri, world_tri_vert_normals, alpha, beta, gamma):
            """Applies the Blinn-Phong shading model to the mesh using its 
            color properties. No support for textures. Has Support for shadow 
            maps.
            """
            o = mesh.diffuse_color
            kd = mesh.kd
            l_color = light.color
            
            p = Vector3.add(Vector3.mul(world_tri[0], alpha), \
                Vector3.add(Vector3.mul(world_tri[1], beta), Vector3.mul(world_tri[2], gamma)))
            
            n = mesh.transform.apply_to_normal(Vector3.add(Vector3.mul(world_tri_vert_normals[0], alpha), \
                Vector3.add(Vector3.mul(world_tri_vert_normals[1], beta), \
                Vector3.mul(world_tri_vert_normals[2], gamma))))

            if isinstance(light, DirectionalLight):
                p_sm = None
                if isinstance(camera, PerspectiveCamera):
                    p_sm = Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[0]), perspective_correction_w(self.camera, ndc_tri[0])), alpha), \
                        Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[1]), perspective_correction_w(self.camera, ndc_tri[1])), beta), \
                        Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[2]), perspective_correction_w(self.camera, ndc_tri[2])), gamma)))
                    p_sm = Vector4.to_Vector3(Vector4.div(p_sm, p_sm[3]))
                else:
                    p_sm = Vector3.add(Vector3.mul(world_tri[0], alpha), Vector3.add(Vector3.mul(world_tri[1], beta), \
                        Vector3.mul(world_tri[2], gamma)))
                unoccluded = shadow_map.check_occlusion(p_sm) if shadow_map is not None else 1

                l = light.transform.apply_to_normal(Vector3.forward())

                phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n) * unoccluded, 0)), np.pi), None, 1)

                d = Vector3.mul(l_color, phi_d)

                v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                h = Vector3.normalize(Vector3.add(l, v))

                i_s = mesh.specular_color
                ks = mesh.ks
                ke = mesh.ke

                phi_s = ks * (max(0, Vector3.dot(n, h)) ** ke) * unoccluded

                s = Vector3.mul(i_s, phi_s)

                a = Vector3.mul(ambient_light, mesh.ka)

                return Vector3.clamp(Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255), None, 255)
            elif isinstance(light, PointLight):
                l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

                phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi), None, 1)

                l_intensity = light.intensity
                l_distance = Vector3.dist(light.transform.get_position(), p)
                id = Vector3.div(Vector3.mul(l_color, l_intensity), (l_distance * l_distance))
                
                d = Vector3.mul(id, phi_d)

                # camera transform get position - p norm
                v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                h = Vector3.normalize(Vector3.add(l, v))

                i_s = mesh.specular_color
                ks = mesh.ks
                ke = mesh.ke
                phi_s = ks * (max(0, Vector3.dot(n, h)) ** ke)

                s = Vector3.mul(i_s, phi_s)

                a = Vector3.mul(ambient_light, mesh.ka)

                return Vector3.clamp(Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255), None, 255)

        def shade_gouraud_vertex(vertex_colors, face, ambient_light, light, camera, mesh, world_tri, world_tri_vert_normals):
            """Calculates Gouraud shading per vertex on the mesh using its 
            color properties. Allows for shadows on mesh but no support for 
            shadow maps. No support for textures.
            """
            for i in range(3):
                if vertex_colors[face[i]] is None:
                    p = world_tri[i]
                    n = world_tri_vert_normals[i]
                    l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

                    o = mesh.diffuse_color
                    kd = mesh.kd
                    phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi), None, 1)

                    l_color = light.color
                    id = None
                    if isinstance(light, DirectionalLight):
                        id = l_color
                    elif isinstance(light, PointLight):
                        l_intensity = light.intensity
                        l_distance = Vector3.dist(light.transform.get_position(), p)
                        id = Vector3.div(Vector3.mul(l_color, l_intensity), (l_distance * l_distance))
                    
                    d = Vector3.mul(id, phi_d)

                    v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                    h = Vector3.normalize(Vector3.add(l, v))

                    i_s = mesh.specular_color
                    ks = mesh.ks
                    ke = mesh.ke
                    phi_s = ks * (max(0, Vector3.dot(n, h)) ** ke)

                    s = Vector3.mul(i_s, phi_s)

                    a = Vector3.mul(ambient_light, mesh.ka)

                    vertex_colors[face[i]] = Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255)

        def shade_gouraud_pixel(vert_color_tri, alpha, beta, gamma):
            """Applies Gouraud shading per pixel to the mesh using its color 
            properties.
            """
            return Vector3.add(Vector3.mul(vert_color_tri[0], alpha), \
                Vector3.add(Vector3.mul(vert_color_tri[1], beta), Vector3.mul(vert_color_tri[2], gamma)))

        def shade_texture(texture_pixels, texture_width, texture_height, uv_tri, alpha, beta, gamma):
            """Applies shading using a texture without pespective correction 
            to the mesh. No support for shadows or shadow maps.
            """
            uv = Vector2.add(Vector2.mul(uv_tri[0], alpha), \
                Vector2.add(Vector2.mul(uv_tri[1], beta), Vector2.mul(uv_tri[2], gamma)))

            x = math.floor((texture_width - 1) * uv[0]) # ? Is subtracting one correct? may be wrong
            y = math.floor((texture_height - 1) * (1 - uv[1]))

            return (texture_pixels[x, y][0], texture_pixels[x, y][1], texture_pixels[x, y][2])

        def shade_texture_correct(camera, texture_pixels, texture_width, texture_height, uv_tri, ndc_tri, \
            alpha, beta, gamma):
            """Applies shading using a texture with pespective correction to 
            the mesh. No support for shadows or shadow maps.
            """
            uv = Vector3.add(Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[0]), perspective_correction_w(camera, ndc_tri[0])), alpha), \
                Vector3.add(Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[1]), perspective_correction_w(camera, ndc_tri[1])), beta), \
                Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[2]), perspective_correction_w(camera, ndc_tri[2])), gamma)))

            x = math.floor((texture_width - 1) * (uv[0] / uv[2]))
            y = math.floor((texture_height - 1) * (1 - (uv[1] / uv[2])))

            return (texture_pixels[x, y][0], texture_pixels[x, y][1], texture_pixels[x, y][2])

        def shade_shadow_map(camera, shadow_map, ndc_tri, world_tri, alpha, beta, gamma):
            """Applies a visual representation of shadow map occlusion to the 
            mesh. Black areas a occluded. White areas are unoccluded.
            """
            p = None
            if isinstance(camera, PerspectiveCamera):
                p = Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[0]), perspective_correction_w(self.camera, ndc_tri[0])), alpha), \
                    Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[1]), perspective_correction_w(self.camera, ndc_tri[1])), beta), \
                    Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[2]), perspective_correction_w(self.camera, ndc_tri[2])), gamma)))
                p = Vector4.to_Vector3(Vector4.div(p, p[3]))
            else:
                p = Vector3.add(Vector3.mul(world_tri[0], alpha), Vector3.add(Vector4.mul(world_tri[1], beta), \
                    Vector3.mul(world_tri[2], gamma)))
            unoccluded = shadow_map.check_occlusion(p)
            return (255 * unoccluded, 255 * unoccluded, 255 * unoccluded)

        def shade_stylized(camera, light, ambient_light, shadow_map, mesh, ndc_tri, world_tri, world_tri_vert_normals, alpha, beta, gamma, x, y):
            """Applies the stylized (non-photorealitic) shading to the mesh 
            using its color properties. No support for textures. Has Support 
            for shadow maps.
            """
            o = mesh.diffuse_color
            kd = mesh.kd
            l_color = light.color
            
            p = Vector3.add(Vector3.mul(world_tri[0], alpha), \
                Vector3.add(Vector3.mul(world_tri[1], beta), Vector3.mul(world_tri[2], gamma)))
            
            n = mesh.transform.apply_to_normal(Vector3.add(Vector3.mul(world_tri_vert_normals[0], alpha), \
                Vector3.add(Vector3.mul(world_tri_vert_normals[1], beta), \
                Vector3.mul(world_tri_vert_normals[2], gamma))))

            if isinstance(light, DirectionalLight):
                p_sm = None
                if isinstance(camera, PerspectiveCamera):
                    p_sm = Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[0]), perspective_correction_w(self.camera, ndc_tri[0])), alpha), \
                        Vector4.add(Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[1]), perspective_correction_w(self.camera, ndc_tri[1])), beta), \
                        Vector4.mul(Vector4.div(Vector3.to_Vector4(world_tri[2]), perspective_correction_w(self.camera, ndc_tri[2])), gamma)))
                    p_sm = Vector4.to_Vector3(Vector4.div(p_sm, p_sm[3]))
                else:
                    p_sm = Vector3.add(Vector3.mul(world_tri[0], alpha), Vector3.add(Vector4.mul(world_tri[1], beta), \
                        Vector3.mul(world_tri[2], gamma)))

                l = light.transform.apply_to_normal(Vector3.forward())
                v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                h = Vector3.normalize(Vector3.add(l, v))
                
                lit = max(Vector3.dot(l, n), 0) # regular fragment lighting
                unoccluded = shadow_map.check_occlusion(p_sm) if shadow_map is not None else 1 # check if area is occluded from the light
                rim = 1 - max(Vector3.dot(v, n), 0) # white around the rim of the object
                rim = 0.5 if rim > 0.8 else 0

                phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * lit * unoccluded), np.pi), None, 1)
                phi_d_no_shadow = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * lit), np.pi), None, 1)

                d = Vector3.mul(l_color, phi_d)
                d_no_shadow = Vector3.mul(l_color, phi_d_no_shadow)

                i_s = mesh.specular_color
                ks = mesh.ks
                ke = mesh.ke

                phi_s = (1 - rim) * ks * (max(0, Vector3.dot(n, h)) ** ke) * unoccluded

                s = Vector3.mul(i_s, phi_s)
                r = Vector3.mul(i_s, rim * lit * unoccluded)
                r_no_shadow = Vector3.mul(i_s, rim * lit)

                a = Vector3.mul(Vector3.mul(ambient_light, mesh.ka), o)
                ar = Vector3.add(a, r)
                ar_no_shadow = Vector3.add(a, r_no_shadow)

                target = Vector3.clamp(Vector3.mul(Vector3.add(Vector3.add(ar, d), s), 255), None, 255)
                color = quantize_color(target)
                target_no_shadow = Vector3.clamp(Vector3.mul(Vector3.add(Vector3.add(ar_no_shadow, d_no_shadow), s), 255), None, 255)
                color_no_shadow = quantize_color(target_no_shadow)
                target_no_s = Vector3.clamp(Vector3.mul(Vector3.add(ar, d), 255), None, 255)
                color_no_s = quantize_color(target_no_s)

                if not np.allclose(color, color_no_shadow):
                    return skew_color(color if pattern.lines(x, y, 4) else color_no_shadow, 1)
                if not np.allclose(color, color_no_s) and np.linalg.norm(s) < 0.2:
                    return skew_color(color if pattern.dots(x, y, 3) else color_no_s, 1)

                target_no_d = Vector3.clamp(Vector3.mul(Vector3.add(ar, s), 255), None, 255)
                color_no_d = quantize_color(target_no_d)

                if not np.allclose(color, color_no_d) and np.linalg.norm(d) < 0.08:
                    return skew_color(color if pattern.dots(x, y, 3) else color_no_d, 1)
                return skew_color(color, 1)
            elif isinstance(light, PointLight): #* No shadow maps for point light
                l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))
                v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                h = Vector3.normalize(Vector3.add(l, v))

                light = max(Vector3.dot(l, n), 0) # regular fragment lighting
                rim = (1 - max(Vector3.dot(v, n), 0)) # white around the rim of the object

                #* To add the rim light, separate the non rim and rim lit sections then add them
                #* since they rim and non_rim are mutually exclusive it is safe to add
                #* rim_lit: rim * lit * unoccluded * rim_color
                #* non_rim_lit: (1 - rim) * lit * unoccluded * non_rim_color
                #* final_color: rim_lit + non_rim_lit

                phi_d = Vector3.clamp(Vector3.div(Vector3.mul(o, kd * lit), np.pi), None, 255)

                l_intensity = light.intensity
                l_distance = Vector3.dist(light.transform.get_position(), p)
                id = Vector3.div(Vector3.mul(l_color, l_intensity), (l_distance * l_distance))
                
                d = Vector3.mul(id, phi_d)

                i_s = mesh.specular_color
                ks = mesh.ks
                ke = mesh.ke
                phi_s = ks * (max(0, Vector3.dot(n, h)) ** ke)

                s = Vector3.mul(i_s, phi_s)

                a = Vector3.mul(Vector3.mul(ambient_light, mesh.ka), o)

                target = Vector3.clamp(Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255), None, 255)
                return skew_color(quantize_color(target), 1)

        def shade_outline(mesh):
            """Applies the outline color to the outline mesh.
            """
            return Vector3.mul(mesh.diffuse_color, 255)

        image_buffer = np.full((self.screen.width, self.screen.height, 3), bg_color)
        depth_buffer = np.full((self.screen.width, self.screen.height), -math.inf, dtype=float)

        def render_pass(image_buffer, depth_buffer, rpass):
            """Renders a specific pass that it is given. Can only render one 
            pass at a time.
            """
            for i in range(len(rpass.meshes)):
                mesh: Mesh = rpass.meshes[i]

                world_verts = [mesh.transform.apply_to_point(p) for p in mesh.verts]
                ndc_verts = [self.camera.project_point(p) for p in world_verts]
                screen_verts = [self.screen.device_to_screen(p) for p in ndc_verts]

                vertex_colors = None
                if rpass.shading == "gouraud":
                    vertex_colors = [None] * len(mesh.verts)

                texture_pixels = None
                texture_width = None
                texture_height = None
                if rpass.shading == "texture" or rpass.shading == "texture-correct":
                    texture_pixels = mesh.texture.load()
                    texture_width = mesh.texture.width
                    texture_height = mesh.texture.height

                for j in range(len(mesh.faces)):
                    face = mesh.faces[j]
                    normal = mesh.normals[j]
                    world_tri_vert_normals = [mesh.vert_normals[face[0]], mesh.vert_normals[face[1]], mesh.vert_normals[face[2]]] if len(mesh.vert_normals) > 0 else None
                    world_tri = [world_verts[face[0]], world_verts[face[1]], world_verts[face[2]]]
                    ndc_tri = [ndc_verts[face[0]], ndc_verts[face[1]], ndc_verts[face[2]]]
                    screen_tri = [screen_verts[face[0]], screen_verts[face[1]], screen_verts[face[2]]]
                    uv_tri = [mesh.uvs[face[0]], mesh.uvs[face[1]], mesh.uvs[face[2]]] if len(mesh.uvs) > 0 else None

                    # cull triangle if normal is facing away from camera
                    cross = Vector3.cross(Vector3.normalize(Vector3.sub(ndc_tri[1], ndc_tri[0])), \
                        Vector3.normalize(Vector3.sub(ndc_tri[2], ndc_tri[0])))
                    if cross[2] < 0:
                        continue

                    # get pixel bounds for which areas of pixels should be drawn
                    bounds = get_pixel_bounds(screen_tri, self.screen.width, self.screen.height)
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

                    vert_color_tri = None
                    if rpass.shading == "gouraud":
                        shade_gouraud_vertex(vertex_colors, face, ambient_light, self.light, self.camera, mesh, \
                            world_tri, world_tri_vert_normals) # face index
                        vert_color_tri = [vertex_colors[face[0]], vertex_colors[face[1]], vertex_colors[face[2]]]

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

                            # render the pixel
                            if rpass.shading == "flat":
                                image_buffer[x, y] = shade_flat(self.light, ambient_light, mesh, \
                                    world_tri, normal, alpha, beta, gamma)
                            elif rpass.shading == "barycentric":
                                image_buffer[x, y] = shade_barycentric(alpha, beta, gamma)
                            elif rpass.shading == "depth":
                                image_buffer[x, y] = shade_depth(depth)
                            elif rpass.shading == "phong-blinn":
                                image_buffer[x, y] = shade_phong_blinn(self.camera, self.light, ambient_light, self.shadow_map, \
                                    mesh, ndc_tri, world_tri, world_tri_vert_normals, alpha, beta, gamma)
                            elif rpass.shading == "gouraud":
                                image_buffer[x, y] = shade_gouraud_pixel(vert_color_tri, alpha, beta, gamma)
                            elif rpass.shading == "texture":
                                image_buffer[x, y] = shade_texture(texture_pixels, texture_width, texture_height, uv_tri, \
                                    alpha, beta, gamma)
                            elif rpass.shading == "texture-correct":
                                image_buffer[x, y] = shade_texture_correct(self.camera, texture_pixels, texture_width, texture_height, uv_tri, \
                                    ndc_tri, alpha, beta, gamma)
                            elif rpass.shading == "shadow-map":
                                image_buffer[x, y] = shade_shadow_map(self.camera, self.shadow_map, ndc_tri, world_tri, alpha, beta, gamma)
                            elif rpass.shading == "stylized":
                                image_buffer[x, y] = shade_stylized(self.camera, self.light, ambient_light, self.shadow_map, \
                                    mesh, ndc_tri, world_tri, world_tri_vert_normals, alpha, beta, gamma, x, y)
                            elif rpass.shading == "outline":
                                image_buffer[x, y] = shade_outline(mesh)

                if rpass.shading == "texture" or rpass.shading == "texture-correct":
                    mesh.texture.close()
                    
        if not isinstance(passes, list):
            passes = [passes]
        for i in range(len(passes)):
            render_pass(image_buffer, depth_buffer, passes[i])
        self.screen.draw(image_buffer)

# Passes are created inside the run_stylized.py and other run scripts
class Pass:
    """Generic type pass object for shading styles that do not require a 
    special implementation. Takes the meshes you would like to apply 
    the specific shading style to.
    """

    def __init__(self, meshes, shading):
        """Constructor for Pass object.
        """
        self.meshes = meshes
        self.shading = shading

class OutlinePass(Pass):
    """Pass object specifically for creating outlines for meshes. Takes the 
    meshes you would like to add outlines to, along arrays of colors and sizes 
    for those outlines. 
    """

    def __init__(self, meshes, mesh_outline_colors, mesh_outline_sizes):
        """Constructor for OutlinePass object.
        """

        def invert_hull(meshes, mesh_outline_sizes):
            """Copies the meshes it is given and applies the inverted hull
            technique to them. This extrudes the mesh out by its vertex 
            normals and flips the normals of the faces.
            """
            inverted_hull_meshes = []
            for i in range(len(meshes)):
                mesh = meshes[i]
                inverted_hull_mesh = mesh.deep_copy()
                
                # move the vertex according to vertex normals
                for j in range(len(inverted_hull_mesh.verts)):
                    inverted_hull_mesh.verts[j] = Vector3.add(inverted_hull_mesh.verts[j], Vector3.mul(inverted_hull_mesh.vert_normals[j], mesh_outline_sizes[i]))
                # reverse vertex face order
                for j in range(len(inverted_hull_mesh.faces)):
                    inverted_hull_mesh.faces[j] = [inverted_hull_mesh.faces[j][2], inverted_hull_mesh.faces[j][1], inverted_hull_mesh.faces[j][0]]
                # calculate the normal
                for j in range(len(inverted_hull_mesh.normals)):
                    a = Vector3.sub(inverted_hull_mesh.verts[inverted_hull_mesh.faces[j][1]], inverted_hull_mesh.verts[inverted_hull_mesh.faces[j][0]])
                    b = Vector3.sub(inverted_hull_mesh.verts[inverted_hull_mesh.faces[j][2]], inverted_hull_mesh.verts[inverted_hull_mesh.faces[j][0]])
                    inverted_hull_mesh.normals[j] = Vector3.normalize(Vector3.cross(a, b))
                # calculate the new vertex normals
                inverted_hull_mesh.calculate_vert_normals()

                inverted_hull_meshes.append(inverted_hull_mesh)

            return inverted_hull_meshes
        
        inverted_hull_meshes = invert_hull(meshes, mesh_outline_sizes)
        for i in range(len(mesh_outline_colors)):
            inverted_hull_meshes[i].diffuse_color = mesh_outline_colors[i]
        super().__init__(inverted_hull_meshes, "outline")
