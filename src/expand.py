import numpy as np

x = np.eye(5)
x[2, 1] = 2
x_ = np.expand_dims(x[:, 1], 0)

print(x)
print(x[:, 1])
print(x_)
