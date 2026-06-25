import numpy as np
import scipy
import matplotlib.pyplot as plt

### a
n = np.arange(0, 16)
x_n = np.cos(np.pi * n / 4)

X_n = scipy.fft.fftshift(scipy.fft.fft(x_n))

### b
x_n_padded32 = np.pad(x_n, (0, 16))

X_n_padded32 = scipy.fft.fftshift(scipy.fft.fft(x_n_padded32))

### c
x_n_padded64 = np.pad(x_n, (0, 48))

X_n_padded64 = scipy.fft.fftshift(scipy.fft.fft(x_n_padded64))

fig, ax = plt.subplots()

ax.stem((np.arange(0, 64)-32)/64*np.pi, np.abs(X_n_padded64), linefmt="green")
ax.stem((np.arange(0, 32)-16)/32*np.pi, np.abs(X_n_padded32), linefmt="red")
ax.stem((np.arange(0, 16)-8)/16*np.pi, np.abs(X_n))
plt.show()

### d
"""
This is again a matter of periodic extensions. The zero-padding operation introduces a constant (and therefore all-pass) signal that spreads frequency power. This can be thought of, in the DTFT of the periodic extension, as a frequency-domain convolution of the line spectra with a sinc (from the time-domain multiplication with a large rectangular pulse), which smears the spectrum. It does, however, have the advantage of increasing spectrum sampling density, and as seen in problem 2, increasing spectrum sampling density makes the DFT more tolerant to frequency noise.
"""
