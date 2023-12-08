import board
import neopixel
from screen import Screen

class LEDScreen(Screen):
    def __init__(self, width, height, brightness):
        self.brightness = brightness
        super().__init__(width, height)

    def show(self):
        pixel_pin = board.D18
        num_pixels = self.width * self.height
        ORDER = neopixel.GRB
        pixels = neopixel.NeoPixel(pixel_pin, num_pixels, brightness=self.brightness, auto_write=False, pixel_order=ORDER)

        for i in range(num_pixels):
            row = i // self.height
            column = (self.width - 1 - (i % self.width)) if (i // self.width) % 2 == 1 else (i % self.width)
            pixels[i] = self.buffer[row, column]

        pixels.show()
        super().show()