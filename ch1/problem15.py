import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
k_1 = 0.2,k_2 = 0.3, k_3 = 0.5, k_4 = 0.7
"""

# Unlike MATLAB, scipy doesn't seem to have a lattice-to-filter-coefficient conversion. So this is an implementation of that, obviously to be refined before it would be included in SciPy according to the contribution guide: [[https://scipy.github.io/devdocs/dev/contributor/contributor_toc.html]]

# NOTE: his directly generates the denominator polynomial of the AP filter, NOT the poles themselves 
def generate_next_poles(reflection_coefficient, previous_poles):
    m = len(previous_poles)
    result = [1]
    result.extend(
        previous_poles[l] + reflection_coefficient * previous_poles[m - l]
        for l in range(1, m)
    )
    result.append(reflection_coefficient)
    return result

def generate_final_poles(reflection_coefficients):
    state = []
    for rc in reflection_coefficients:
        state = generate_next_poles(rc, state)
    return state

def generate_previous_poles(reflection_coefficient, forward_poles):
    m = len(forward_poles)-1
    result = [1]
    result.extend(
        (forward_poles[l] - reflection_coefficient*np.conj(forward_poles[m-l]))/(1 - np.abs(reflection_coefficient)**2)
        for l in range(1, m)
    )
    return result

def generate_reflections(pole_coefficients):
    result = []
    state = pole_coefficients
    for i in range(len(pole_coefficients)):
        current_coefficient = state[-1]
        result.insert(0, current_coefficient)
        state = generate_previous_poles(current_coefficient, state)
    return result

reflection_coefficients = [0.2, 0.3, 0.5, 0.7]
pole_coefficients = generate_final_poles(reflection_coefficients)

omega, mag = scipy.signal.freqz(1, pole_coefficients)

plt.plot(omega, mag)
plt.show()
