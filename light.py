from transform import Transform

class PointLight:
    def __init__(self, intensity, color):
        self.intensity = intensity
        self.color = color
        self.transform = Transform()