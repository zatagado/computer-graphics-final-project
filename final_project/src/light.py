from transform import Transform

class DirectionalLight:
    """A directional light in space that is constant across the entire scene.

    Attributes
    ----------
    transform : Transform
        A Transform object exposed to set the orientation (position and 
        rotation) of the camera. This should default to represent a position 
        of (0, 0, 0) and no rotation.

    color : list()
        A 3 element (RGB) array representing the color of the light source 
        using values between 0.0 and 1.0.
    """

    def __init__(self, color): # we don't really care to implement the intensity since there is no falloff
        self.color = color
        self.transform = Transform()

class PointLight:
    """A point light in space with light falloff.

    Attributes
    ----------
    transform : Transform
        A Transform object exposed to set the orientation (position and 
        rotation) of the camera. This should default to represent a position 
        of (0, 0, 0) and no rotation.

    intensity : float
        A scalar value representing the intensity, or brightness of the light 
        source.

    color : list()
        A 3 element (RGB) array representing the color of the light source 
        using values between 0.0 and 1.0.
    """
    def __init__(self, intensity, color):
        self.intensity = intensity
        self.color = color
        self.transform = Transform()