import numpy as np
import scipy
import matplotlib.pyplot as plt

### a
x_n = np.array([1, 2, 3, 4, 3, 2, 1])
h_n = np.array([-1, 0, 1])

X_n = np.array([[1, 0, 0],
       [2, 1, 0],
       [3, 2, 1],
       [4, 3, 2],
       [3, 4, 3],
       [2, 3, 4],
       [1, 2, 3],
       [0, 1, 2],
       [0, 0, 1]
        ])

### b
def conv_toeplitz(x, h):
    x_toep = scipy.linalg.toeplitz(np.pad(x, (len(h), len(h) - 1)))[len(h):, :len(h)]
    return x_toep @ h

### c
print(np.max(conv_toeplitz(x_n, h_n) - X_n @ h_n))
print(max(conv_toeplitz(x_n, [1, 0, 0]) - np.pad(x_n, (0, 2))))
print(max(conv_toeplitz(x_n, np.pad([1], (0, 10))) - np.pad(x_n, (0, 10))))
print(max(conv_toeplitz(x_n, [1, 1, 1]) - [1, 3, 6, 9, 10, 9, 6, 3, 1]))
