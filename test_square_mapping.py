from src.grid import square_from_point

tests = [
    (50, 750),   # a1
    (450, 350),  # e4
    (750, 50),   # h8
]

for x, y in tests:
    print((x, y), "->", square_from_point(x, y))
