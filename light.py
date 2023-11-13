from transform import Transform

class DirectionalLight:
    def __init__(self, color):
        self.color = color
        self.transform = Transform()

class PointLight:
    def __init__(self, intensity, color):
        self.intensity = intensity
        self.color = color
        self.transform = Transform()