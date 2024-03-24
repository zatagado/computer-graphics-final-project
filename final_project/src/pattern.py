import math

def checker(x, y, size):
    return ((x // size) % 2 == 0) ^ ((y // size) % 2 == 0)

def _dots(x, y, size):
    period_x = size * 2
    period_y = period_x * math.sqrt(3)
    cell_x = x % period_x
    cell_y = y % period_y
    radius = size / math.sqrt(math.pi)
    return (cell_x - size) ** 2 + (cell_y - size) ** 2 <= radius ** 2

def dots(x, y, size):
    return _dots(x, y, size) or _dots(x + size, y + size * math.sqrt(3), size)

def lines(x, y, size):
    return (x - 2 * y) % (size * 2) >= size
