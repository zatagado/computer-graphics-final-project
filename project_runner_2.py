import numpy as np

from screen import Screen
from camera import PerspectiveCamera,OrthoCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight, DirectionalLight
from shadow_map import ShadowMap
from render_math import Vector3

if __name__ == '__main__':
    screen = Screen(500,500)

    # camera = OrthoCamera(6.0, -6.0, -6.0, 6.0, -1.0, -20) #* shadows work on orthographic camera
    camera = PerspectiveCamera(1.0, -1.0, -1.0, 1.0, -1.0, -20.0)
    camera.transform.set_position(0, 10, 1)
    camera.transform.set_rotation_towards([0, -1, -0.3])

    mesh_1 = Mesh.from_stl("unit_sphere.stl", np.array([1.0, 0.6, 0.0]),\
        np.array([1.0, 1.0, 1.0]),0.05,0.8,0.2,100)
    mesh_1.transform.set_position(0.5, 0, 1)

    mesh_2 = Mesh.from_stl("plane.stl", np.array([0.9, 0.9, 0.9]), \
        np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_2.transform.set_position(0, 0, -2.5)

    light = DirectionalLight(np.array([1, 1, 1]))
    light.transform.set_rotation_towards(Vector3.normalize([1, 1, -1]))

    shadow_map = ShadowMap([mesh_1, mesh_2], light, OrthoCamera(7, -7, -7, 7, 5.0, -20), (screen.width, screen.height), 0.1) # TODO changing
    
    # p = [0, 0, -2.5]
    # print(shadow_map.orthoCamera.project_point(p))
    # print(shadow_map.device_to_screen(shadow_map.orthoCamera.project_point(p)))
    # print(camera.project_point(p))
    # print(screen.device_to_screen(camera.project_point(p)))

    #* Grouping for rendering from camera pov
    renderer = Renderer(screen, camera, [mesh_1, mesh_2], light, shadow_map)
    renderer.render("shadow-map", [80, 80, 80], [0.2, 0.2, 0.2]) # TODO find out the issue with the ambient light when color is changed

    screen.show()