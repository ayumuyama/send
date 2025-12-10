import numpy as np

A = np.array([[1, 2, 3, 4]])
B = np.array([[7]])
print(f'"A: "{A.shape}')
print(f'"B: "{B.shape}')
C = np.dot(B, A)
print(f'"C: "{C}')
print(f'"C: "{C.shape}')
