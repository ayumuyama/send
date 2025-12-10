import numpy as np

x = np.where(np.random.rand(2, 3) < 0.5, 1, 0)
print(x)
print(x.shape)
x1 = x[:, -1]
print(x1)
print(x1.shape)