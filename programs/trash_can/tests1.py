import numpy as np

# Trying to understand how reshape works
x = np.linspace(0, 1, 2)
y = np.linspace(0, 2, 3)
z = np.linspace(0, 3, 4)

X,Y,Z = np.meshgrid(x, y, z)

grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
print(grid)
v = np.linspace(1, 24, 24)
v = v.reshape(X.shape)
print(v)

# slice it along the first axis
print(v[0, :, :])