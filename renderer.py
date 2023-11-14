from screen import Screen
from mesh import Mesh
from render_math import Vector2, Vector3, Vector4, Shader
from light import PointLight, DirectionalLight
import numpy as np
import math

class Renderer:
    def __init__(self, screen: Screen, camera, meshes: list, light):
        self.screen = screen
        self.camera = camera
        self.meshes = meshes
        self.light = light

    def render(self, shading, bg_color, ambient_light):
        def get_pixel_bounds(screen_coords_verts, screen_width, screen_height):
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

        def shade_flat(ambient_light, light, mesh, world_tri, normal, alpha, beta, gamma):
            p = Vector3.add(Vector3.mul(world_tri[0], alpha), \
                Vector3.add(Vector3.mul(world_tri[1], beta), Vector3.mul(world_tri[2], gamma)))
            
            n = mesh.transform.apply_to_normal(normal)
            l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

            o = mesh.diffuse_color
            kd = mesh.kd
            phi_d = Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi)
            phi_d = np.array([min(1, phi_d[0]), min(1, phi_d[1]), min(1, phi_d[2])])


            l_color = light.color
            l_intensity = light.intensity
            l_distance = Vector3.dist(light.transform.get_position(), p)
            id = Vector3.div(Vector3.mul(l_color, l_intensity), (l_distance * l_distance))
            
            d = Vector3.mul(id, phi_d)
            a = Vector3.mul(ambient_light, mesh.ka)

            final = Vector3.mul(Vector3.add(a, d), 255)
            return (min(255, final[0]), min(255, final[1]), min(255, final[2]))

        def shade_barycentric(alpha, beta, gamma):
            return (255 * alpha, 255 * beta, 255 * gamma)
        
        def shade_depth(depth):
            depth_norm = (depth + 1) / 2
            return (255 * depth_norm, 255 * depth_norm, 255 * depth_norm)
        
        def shade_phong_blinn(ambient_light, light, camera, mesh, world_tri, world_tri_vert_normals, alpha, beta, gamma):
            o = mesh.diffuse_color
            kd = mesh.kd
            l_color = light.color
            
            p = Vector3.add(Vector3.mul(world_tri[0], alpha), \
                Vector3.add(Vector3.mul(world_tri[1], beta), Vector3.mul(world_tri[2], gamma)))
            
            n = mesh.transform.apply_to_normal(Vector3.add(Vector3.mul(world_tri_vert_normals[0], alpha), \
                Vector3.add(Vector3.mul(world_tri_vert_normals[1], beta), \
                Vector3.mul(world_tri_vert_normals[2], gamma))))

            # TODO check if the directional light calculation is correct
            if isinstance(light, DirectionalLight):
                l = light.transform.apply_to_normal(Vector3.forward())

                phi_d = Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi)
                phi_d = np.array([min(1, phi_d[0]), min(1, phi_d[1]), min(1, phi_d[2])], dtype=float)

                d = Vector3.mul(l_color, phi_d)

                v = Vector3.normalize(Vector3.sub(camera.transform.get_position(), p))
                h = Vector3.normalize(Vector3.add(l, v))

                i_s = mesh.specular_color
                ks = mesh.ks
                ke = mesh.ke
                phi_s = ks * (max(0, Vector3.dot(n, h)) ** ke)

                s = Vector3.mul(i_s, phi_s) # TODO looks slightly off on depth 2

                a = Vector3.mul(ambient_light, mesh.ka)

                final = Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255)
                return (min(255, final[0]), min(255, final[1]), min(255, final[2]))
            elif isinstance(light, PointLight):
                l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

                phi_d = Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi)
                phi_d = np.array([min(1, phi_d[0]), min(1, phi_d[1]), min(1, phi_d[2])], dtype=float)

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

                s = Vector3.mul(i_s, phi_s) # TODO looks slightly off on depth 2

                a = Vector3.mul(ambient_light, mesh.ka)

                final = Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255)
                return (min(255, final[0]), min(255, final[1]), min(255, final[2]))

        def shade_gouraud_vertex(vertex_colors, face, ambient_light, light, camera, mesh, world_tri, world_tri_vert_normals):
            for i in range(3):
                if vertex_colors[face[i]] is None:
                    p = world_tri[i]
                    n = world_tri_vert_normals[i]
                    l = Vector3.normalize(Vector3.sub(light.transform.get_position(), p))

                    o = mesh.diffuse_color
                    kd = mesh.kd
                    phi_d = Vector3.div(Vector3.mul(o, kd * max(Vector3.dot(l, n), 0)), np.pi)
                    phi_d = np.array([min(1, phi_d[0]), min(1, phi_d[1]), min(1, phi_d[2])], dtype=float)

                    l_color = light.color
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

                    s = Vector3.mul(i_s, phi_s) # TODO looks slightly off on depth 2

                    a = Vector3.mul(ambient_light, mesh.ka)

                    vertex_colors[face[i]] = Vector3.mul(Vector3.add(Vector3.add(a, d), s), 255)

        def shade_gouraud_pixel(vert_color_tri, alpha, beta, gamma):
            return Vector3.add(Vector3.mul(vert_color_tri[0], alpha), \
                Vector3.add(Vector3.mul(vert_color_tri[1], beta), Vector3.mul(vert_color_tri[2], gamma)))

        def shade_texture(texture_pixels, texture_width, texture_height, uv_tri, alpha, beta, gamma):
            uv = Vector2.add(Vector2.mul(uv_tri[0], alpha), \
                Vector2.add(Vector2.mul(uv_tri[1], beta), Vector2.mul(uv_tri[2], gamma)))

            x = math.floor((texture_width - 1) * uv[0]) # ? Is subtracting one correct? may be wrong
            y = math.floor((texture_height - 1) * (1 - uv[1]))

            return (texture_pixels[x, y][0], texture_pixels[x, y][1], texture_pixels[x, y][2])

        def shade_texture_correct(camera, texture_pixels, texture_width, texture_height, uv_tri, ndc_tri, \
            alpha, beta, gamma):
            def get_w(camera, p):
                p_copy = np.array(p, dtype=float)
                p_copy[1], p_copy[2] = p_copy[2], p_copy[1]
                p_persp = np.matmul(camera.inverse_ortho_transform, Vector4.to_vertical(Vector3.to_Vector4(p_copy)))
                y = (camera.far * camera.near) / ((camera.near + camera.far) - p_persp[1])
                return y

            # TODO something is wrong, seemed better with ndc_tri[1]
            # ? does it matter if we use the depth pre inverse perspective or not?
            uv = Vector3.add(Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[0]), get_w(camera, ndc_tri[0])), alpha), \
                Vector3.add(Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[1]), get_w(camera, ndc_tri[1])), beta), \
                Vector3.mul(Vector3.div(Vector2.to_Vector3(uv_tri[2]), get_w(camera, ndc_tri[2])), gamma)))

            x = math.floor((texture_width - 1) * (uv[0] / uv[2]))
            y = math.floor((texture_height - 1) * (1 - (uv[1] / uv[2])))

            return (texture_pixels[x, y][0], texture_pixels[x, y][1], texture_pixels[x, y][2])

        image_buffer = np.full((self.screen.width, self.screen.height, 3), bg_color)
        depth_buffer = np.full((self.screen.width, self.screen.height), -math.inf, dtype=float)

        for i in range(len(self.meshes)):
            mesh: Mesh = self.meshes[i]

            world_verts = [mesh.transform.apply_to_point(p) for p in mesh.verts]
            ndc_verts = [self.camera.project_point(p) for p in world_verts]
            screen_verts = [self.screen.device_to_screen(p) for p in ndc_verts]
            
            vertex_colors = [None] * len(mesh.verts) if shading == "gouraud" else None

            texture_pixels = None
            texture_width = None
            texture_height = None
            if shading == "texture" or shading == "texture-correct":
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
                if shading == "gouraud":
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
                        if shading == "flat":
                            image_buffer[x, y] = shade_flat(ambient_light, self.light, mesh, \
                                world_tri, normal, alpha, beta, gamma)
                        elif shading == "barycentric":
                            image_buffer[x, y] = shade_barycentric(alpha, beta, gamma)
                        elif shading == "depth":
                            image_buffer[x, y] = shade_depth(depth)
                        elif shading == "phong-blinn":
                            image_buffer[x, y] = shade_phong_blinn(ambient_light, self.light, self.camera, mesh, \
                                world_tri, world_tri_vert_normals, alpha, beta, gamma)
                        elif shading == "gouraud":
                            image_buffer[x, y] = shade_gouraud_pixel(vert_color_tri, alpha, beta, gamma)
                        elif shading == "texture":
                            image_buffer[x, y] = shade_texture(texture_pixels, texture_width, texture_height, uv_tri, \
                                alpha, beta, gamma)
                        elif shading == "texture-correct":
                            image_buffer[x, y] = shade_texture_correct(self.camera, texture_pixels, texture_width, texture_height, uv_tri, \
                                ndc_tri, alpha, beta, gamma)

            if shading == "texture" or shading == "texture-correct":
                mesh.texture.close()

        self.screen.draw(image_buffer)