import numpy as np
import scipy
import matplotlib.pyplot as plt

### a
"""
Fully analytic. The convolution of the two signals is equal to the product of their Z transforms with an inverse Z transform.

1/(1 - 0.9z^(-1))^2 -> (n+1)C(1) 0.9^n u[n] = (n+1) (0.9)^n u[n] for the causal version
"""

n = np.arange(0, 101)
x_n = 0.9 ** n
x_n_convolved = np.multiply((n+1), 0.9 ** n)

fig, ax = plt.subplots()

ax.plot(x_n_convolved)

### b
x_n51 = x_n[:51]
x_n51_convolved = np.convolve(x_n51, x_n51)
ax.plot(x_n51_convolved, color="red")

### c
""" y[n] = x[n] + 0.9*y[n-1] """
x_n_filtered = scipy.signal.lfilter([1], [1, -0.9], x_n)
ax.plot(x_n_filtered, color="green")
plt.show()

### d
fig, ax = plt.subplots()
ax.plot((x_n_convolved - x_n51_convolved) ** 2, color="red")
ax.plot((x_n_convolved - x_n_filtered) ** 2, color="green")
plt.show()

"""
The convolve function has the largest error, likely due to the truncation. An infinite-length sequence is best handled analytically or with the filter function, which I believe in Python solves things in the frequency domain anyway.
"""
