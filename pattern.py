checker_size = 6

def checker(x, y):
    return ((x // checker_size) % 2 == 0) ^ ((y // checker_size) % 2 == 0)
