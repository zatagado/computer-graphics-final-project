import numpy as np

from screen import Screen
from camera import PerspectiveCamera,OrthoCamera
from mesh import Mesh
from renderer import Renderer
from light import PointLight, DirectionalLight
from render_math import Vector3



if __name__ == '__main__':
    screen = Screen(500,500)

    camera = PerspectiveCamera(1.0, -1.0, -1.0, 1.0, -1.0, -20)
    camera.transform.set_position(0, 4, 0)

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
    mesh_4.transform.set_position(0, -2, -2)

    light = DirectionalLight(np.array([1, 1, 1]))
    light.transform.set_rotation(0, -45, -90)
    # sim_pos = Vector3.mul(light.transform.apply_to_normal(Vector3.forward()), -10)

    lightSim = PerspectiveCamera(1.0, -1.0, -1.0, 1.0, -1.0, 20)
    # lightSim.transform.set_rotation(0, -45, -90)
    # lightSim.transform.set_rotation(0, 0, -45)
    # lightSim.transform.set_axis_rotation(np.array([1, 1, 1], dtype=float), 0)
    # lightSim.transform.set_position(sim_pos)
    lightSim.transform.set_position(0, 5, 0)
    # lightSim.transform.set_rotation_towards(Vector3.negate(Vector3.normalize([5, 5, 5])))
    lightSim.transform.set_rotation_towards(Vector3.forward())

    # light = PointLight(50.0, np.array([1, 1, 1]))
    # light.transform.set_position(-4, 4, -3)

    # renderer = Renderer(screen, camera, [mesh_1, mesh_2, mesh_3, mesh_4], light)
    renderer = Renderer(screen, lightSim, [mesh_1, mesh_2, mesh_3, mesh_4], light)
    renderer.render("phong-blinn",[80,80,80], [0.2, 0.2, 0.2]) # TODO find out the issue witht the ambient light when color is changed

    screen.show()