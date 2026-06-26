import numpy as np
import scipy
import matplotlib.pyplot as plt

### a
"""
H_{min}(z) = (1 - 0.25z^{-1})(1 + 0.5z^{-1})
H_{max}(z) = (0.25 - z^{-1})(0.5 + z^{-1})
H_{mix}(z) = (1 - 0.25z^{-1})(0.5 + z^{-1})

These are all-zero systems.
H_{min} zeroes: 0.25, -0.5
H_{max} zeroes: 4, -2
H_{mix} zeroes: 0.25, -2

So calculate the phase response and derive for the group delay plot. These are all trivially stable, so we can calculate the phase response analytically.

Relevant phase responses: 
(1 - pz^{-1}) -> arctan(p*sin(\omega)/(1 - p*cos(\omega)))
(p - z^{-1}) -> arctan((1/p)*sin(\omega)/(1 - (1/p)*cos(\omega))) 

Relevant group delays:
-d/d\omega arctan(sin(\omega)/(1 - p*cos(\omega))) = -(p*cos(\omega) - p^2)/(1 - 2*p*cos(\omega) + p^2) 
"""
omega = np.linspace(0, np.pi, 1000)

def gd_zero(zero):
    return -((zero)*np.cos(omega) - (zero)**2)/(1 - 2*(zero)*np.cos(omega) + (zero)**2)

H_min_gd = gd_zero(0.25) + gd_zero(-0.5)
H_max_gd = gd_zero(4) + gd_zero(-2)
H_mix_gd = gd_zero(0.25) + gd_zero(-2)

fig, ax = plt.subplots()
ax.plot(omega, H_min_gd)
ax.plot(omega, H_max_gd, color="red")
ax.plot(omega, H_mix_gd, color="green")
plt.show()

### b
"""
Rough Heuristic:
It is known that the transfer function of any causal PZ system, H, can be expressed in terms of the product of a minium-phase system and an all-pass system. Note that the minimum-phase system has the same magnitude response as H, since the all-pass system has a magnitude response of 1 across all frequencies. We have seen in problem 2.7 that all-pass systems exhibit monotonically decreasing (or, for the trivial identity all-pass, non-increasing) phase responses (and therefore positive group delay and additional lag) across all frequencies under Nyquist. The addition of this positive quantity means that the minimum-phase system must have a smaller group delay than system H.
"""




