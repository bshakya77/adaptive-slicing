import math

# Simulate your conversion function on that single line
coords = [0.109375, 0.34375, 0.671875, 0.34375, 0.671875, 0, 0.109375, 0]
xs = coords[0::2]; ys = coords[1::2]
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
xc = (xmin + xmax)/2
yc = (ymin + ymax)/2
w  = xmax - xmin
h  = ymax - ymin

# Assertions (with a tiny tolerance) #Replace the values with the converted values YOLO format
assert math.isclose(xc, 0.390625, rel_tol=1e-6)  
assert math.isclose(yc, 0.171875, rel_tol=1e-6)
assert math.isclose(w,  0.562500, rel_tol=1e-6)
assert math.isclose(h,  0.343750, rel_tol=1e-6)
print("Unit test passed!")


