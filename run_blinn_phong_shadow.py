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

    camera = PerspectiveCamera(1.0, -1.0, -1.0, 1.0, -1.0, -20)
    camera.transform.set_position(0, 8, 1)

    mesh_1 = Mesh.from_stl("suzanne.stl", np.array([1.0, 0.0, 1.0]),\
        np.array([1.0, 1.0, 1.0]),0.05,1.0,0.2,100)
    mesh_1.transform.set_rotation(15, 0, 35)
    mesh_1.transform.set_position(1,-1,0)

    mesh_2 = Mesh.from_stl("unit_cube.stl", np.array([0.6, 0.0, 1.0]),\
        np.array([1.0, 1.0, 1.0]),0.05,1.0,0.2,100)
    mesh_2.transform.set_position(-0.25, 1.5,-0.4)
    mesh_2.transform.set_rotation(0, 10, 0)

    mesh_3 = Mesh.from_stl("unit_sphere.stl", np.array([1.0, 0.6, 0.0]),\
        np.array([1.0, 1.0, 1.0]),0.05,0.8,0.2,100)
    mesh_3.transform.set_position(-0.4,0,0.75)

    mesh_4 = Mesh.from_stl("plane.stl", np.array([0.9, 0.9, 0.9]), \
        np.array([1.0, 1.0, 1.0]), 0.05, 1.0, 0.2, 100)
    mesh_4.transform.set_position(0, 0, -2.5)

    light = DirectionalLight(np.array([1, 1, 1]))
    light.transform.set_rotation_towards([-0.7, -0.1, -1])

    shadow_map = ShadowMap([mesh_1, mesh_2, mesh_3, mesh_4], light, OrthoCamera(7, -7, -7, 7, 5.0, -20), (screen.width, screen.height), 0.1)
    
    #* Grouping for rendering from camera pov
    renderer = Renderer(screen, camera, [mesh_1, mesh_2, mesh_3, mesh_4], light, shadow_map)
    renderer.render("phong-blinn", [80, 80, 80], [0.2, 0.2, 0.2]) # TODO find out the issue with the ambient light when color is changed

    screen.show()