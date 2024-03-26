import stl
import math
import numpy as np
from render_math import Vector2, Vector3
from transform import Transform
from PIL import Image

class Mesh:
    """Mesh class defines a 3D mesh, which can be created from manually adding 
    vertices and triangles, or loaded from a .stl file.

    Attributes
    ----------
    verts : list()
        List of 3D vertices <x,y,z> for the mesh.

    faces : list()
        List of triangle faces for the mesh, with each face defined as a list 
        of 3 vertex indicies into verts in counter clockwise ordering.

    normals : list()
        List of 3D face normals for the mesh. The elements of this list 
        correspond to the same triangles defined in faces.

    vertex_normals : list()
        The calculated average of all faces connected to each vertex.

    uvs : list()
        Normalized texture coordinates for each vertex.

    texture : Image
        The texture image to be sampled using u,v coordinates.
    """

    def __init__(self, diffuse_color=None, specular_color=None, 
        ka=None, kd=None, ks=None, ke=None):
        """The constructor takes diffuse and specular color as an 3 element np 
        array with all three values between 0.0 and 1.0, as well as material 
        properties ka, kd, ks, and ke.
        """
        self.verts = []
        self.faces = []
        self.normals = []
        self.vert_normals = []
        self.uvs = []
        self.texture = None
        self.transform = Transform()
        self.bounding_box_min = np.array([math.nan, math.nan, math.nan], dtype=float)
        self.bounding_box_max = np.array([math.nan, math.nan, math.nan], dtype=float)
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.ka = ka 
        self.kd = kd
        self.ks = ks
        self.ke = ke

    def calculate_bounding_box(self):
        """Calculates the bounding box of the mesh.  
        """
        self.bounding_box_min = np.array([self.verts[0][0], \
            self.verts[0][1], self.verts[0][2]], dtype=float)
        self.bounding_box_max = np.array([self.verts[0][0], \
            self.verts[0][1], self.verts[0][2]], dtype=float)

        for vertex in self.verts:
            if vertex[0] > self.bounding_box_max[0]:
                self.bounding_box_max[0] = vertex[0]
            elif vertex[0] < self.bounding_box_min[0]:
                self.bounding_box_min[0] = vertex[0]

            if vertex[1] > self.bounding_box_max[1]:
                self.bounding_box_max[1] = vertex[1]
            elif vertex[1] < self.bounding_box_min[1]:
                self.bounding_box_min[1] = vertex[1]

            if vertex[2] > self.bounding_box_max[2]:
                self.bounding_box_max[2] = vertex[2]
            elif vertex[2] < self.bounding_box_min[2]:
                self.bounding_box_min[2] = vertex[2]

    def calculate_vert_normals(self):
        """Calculates the normal direction for each vertex in the mesh.
        """
        tri_for_verts = []
        for i in range(len(self.verts)):
            tri_for_verts.append([])

        for i in range(len(self.faces)):
            for j in range(len(self.faces[i])):
                # append that normal to each vertex in the face
                tri_for_verts[self.faces[i][j]].append(self.normals[i])

        self.vert_normals = [None] * len(self.verts)
        for i in range(len(tri_for_verts)):
            sum = np.array([0, 0, 0], dtype=float)
            for j in range(len(tri_for_verts[i])):
                sum = Vector3.add(sum, tri_for_verts[i][j])
            self.vert_normals[i] = Vector3.normalize(sum)

    @staticmethod
    def from_stl(stl_path, diffuse_color, specular_color, ka, kd, ks, ke):
        """This static method takes an stl file as input, initializes an empty 
        Mesh object using the input material properties diffuse_color, 
        specular_color, ka, kd, ks, ke and populates the verts, faces, and 
        normals member variables. The method returns the populated Mesh object.
        """
        def index_of(arr: list, elem):
            """Finds the index of an element in a list.
            """
            for i in range(len(arr)):
                if Vector3.equals(arr[i], elem):
                    return i
            return -1
        
        new_mesh = Mesh(diffuse_color, specular_color, ka, kd, ks, ke)
        # parse from the path of the stl
        mesh_data = stl.mesh.Mesh.from_file(stl_path)
        
        for i in range(len(mesh_data.vectors)):
            new_mesh.faces.append([])

            for j in range(len(mesh_data.vectors[i])):
                vector = mesh_data.vectors[i][j]
                k = index_of(new_mesh.verts, vector)
                if k == -1:
                    new_mesh.verts.append(vector)
                    new_mesh.faces[i].append(len(new_mesh.verts) - 1)
                else:
                    new_mesh.faces[i].append(k)

            a = Vector3.sub(new_mesh.verts[new_mesh.faces[i][1]], new_mesh.verts[new_mesh.faces[i][0]])
            b = Vector3.sub(new_mesh.verts[new_mesh.faces[i][2]], new_mesh.verts[new_mesh.faces[i][0]])
            new_mesh.normals.append(Vector3.normalize(Vector3.cross(a, b))) #* normalization could cause future issues
        
        new_mesh.calculate_vert_normals()
        new_mesh.calculate_bounding_box()

        new_mesh.diffuse_color = diffuse_color
        new_mesh.specular_color = specular_color
        new_mesh.ka = ka
        new_mesh.kd = kd
        new_mesh.ks = ks
        new_mesh.ke = ke

        return new_mesh

    def load_texture(self, img_path):
        """Load the image file img_path with PIL.Image.open() to populate the 
        texture variable.
        """
        self.texture = Image.open(img_path, "r")

    def sphere_uvs(self):
        """Populates uvs coordinates using cart2sph texture mapping.
        """
        #z is 'up', theta is azimith angle, phi is elevation
        #used for texture coordinates of sphere, so it only returns theta and phi angle
        def cart2sph(v):
            """A texture mapping function using cartesian to spherical 
            conversion to return u,v coordinates.
            """
            x = v[0]
            y = v[1]
            z = v[2]

            theta = np.arctan2(y,x)
            phi = np.arctan2(z,np.sqrt(x**2 + y**2))
            return np.array([theta, phi])
        
        uvs = []
        verts = self.verts

        #loop over each vertex
        for v in verts:
            #convert cartesian to spherical
            uv = cart2sph(v)

            #convert theta and phi to u and v by normalizing angles
            uv[0] += np.pi
            uv[1] += np.pi/2.0

            uv[0] /= 2.0*np.pi
            uv[1] /= np.pi

            uvs.append(uv)

        self.uvs = uvs

        return uvs

    @staticmethod
    def textured_quad():
        """A static function that returns a populated Mesh object that 
        represents a two-triangle quad for demonstrating perspective correct 
        texture coordinates.
        """
        mesh = Mesh() 

        mesh.verts = [np.array([0.4, 0.5, -0.5]),
            np.array([-0.4, 0.5, -0.5]),
            np.array([0.5, -0.5, 0.4]),
            np.array([-0.4, -0.55, 0.4])
            ]

        mesh.faces = [[0, 1, 2], [3, 2, 1]]

        normals = []
        for face in mesh.faces:
            a = Vector3.sub(mesh.verts[face[1]], mesh.verts[face[0]])
            b = Vector3.sub(mesh.verts[face[2]], mesh.verts[face[0]])
            n = Vector3.cross(a, b)

            normals.append(Vector3.normalize(n))

        mesh.normals = normals 

        mesh.uvs = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), \
            np.array([0.0, 1.0]), np.array([1.0, 1.0])]

        return mesh
    
    def deep_copy(self):
        """Creates a deep copy of the mesh object.
        """
        new_mesh = Mesh(self.diffuse_color, self.specular_color, self.ka, self.kd, self.ks, self.ke)
        new_mesh.verts = [Vector3.copy(self.verts[a]) for a in range(len(self.verts))]
        new_mesh.faces = [Vector3.copy(self.faces[a]) for a in range(len(self.faces))]
        new_mesh.normals = [Vector3.copy(self.normals[a]) for a in range(len(self.normals))]
        new_mesh.vert_normals = [Vector3.copy(self.vert_normals[a]) for a in range(len(self.vert_normals))]
        new_mesh.uvs = [Vector2.copy(self.uvs[a]) for a in range(len(self.uvs))]
        new_mesh.texture = self.texture
        new_mesh.transform.model_matrix = np.copy(self.transform.model_matrix)
        new_mesh.bounding_box_min = np.copy(self.bounding_box_min)
        new_mesh.bounding_box_max = np.copy(self.bounding_box_max)
        return new_mesh