import numpy as np

from screen import Screen, LEDMatrix
from camera import PerspectiveCamera,OrthoCamera
from mesh import Mesh
from renderer import Renderer, Pass
from light import PointLight, DirectionalLight
from shadow_map import ShadowMap
from render_math import Vector3

# sudo python3 run_led.py
if __name__ == '__main__':
    screen = LEDMatrix(0.02)

    camera = PerspectiveCamera(1.0, -1.0, -1.0, 1.0, -1.0, -20)
    camera.transform.set_position(0, 3, 0)

    mesh_2 = Mesh.from_stl("unit_cube.stl", np.array([0.6, 0.0, 1.0]),\
        np.array([1.0, 1.0, 1.0]),0.05,1.0,0.2,100)
    mesh_2.transform.set_position(0.0, 1.5, 0.0)
    mesh_2.transform.set_rotation(0, 10, 0)

    light = DirectionalLight(np.array([1, 1, 1]))
    light.transform.set_rotation_towards([-0.7, -0.1, -1])
    
    #* Grouping for rendering from camera pov
    renderer = Renderer(screen, camera, light, None)
    renderer.render(Pass([mesh_2], "barycentric"), [80, 80, 80], [0.2, 0.2, 0.2]) 

    screen.show()