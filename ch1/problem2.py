import numpy as np
import scipy
import matplotlib.pyplot as plt

def gen_S(omega, omega_0, N):
    return np.divide(np.sin( (omega - omega_0) * N/2 ), 
              np.sin( (omega - omega_0) / 2 ))

def gen_S_prime(omega, omega_0, N):
    return np.divide(np.sin( (omega + omega_0) * N/2 ), 
              np.sin( (omega + omega_0) / 2 ))

def gen_real_term(omega, omega_0, N, A):
    S = gen_S(omega, omega_0, N)
    S_prime = gen_S_prime(omega, omega_0, N)

    return np.multiply((A/2) * np.cos( (omega - omega_0)  * (N-1)/2 ), S) + np.multiply((A/2) * np.cos( (omega + omega_0) * (N-1)/2 ), S_prime)

def gen_imag_term(omega, omega_0, N, A):
    S = gen_S(omega, omega_0, N)
    S_prime = gen_S_prime(omega, omega_0, N)

    return np.multiply((-A/2) * np.sin( (omega - omega_0)  * (N-1)/2 ), S) + np.multiply((-A/2) * np.sin( (omega + omega_0)  * (N-1)/2 ), S_prime)

### a

"""
Entirely analytic, the solution is:
Real Term:
(A/2) cos(
    [\omega - \omega_0][N-1]/2
    )*S
+ (A/2) cos(
    [\omega + \omega_0][N-1]/2
    )*S'

Imaginary Term:
(-A/2) sin([\omega - \omega_0][N-1]/2])*S
(-A/2) sin([\omega + \omega_0][N-1]/2)*S'

Where S = \sin([\omega - omega_0]*N/2)/\sin([\omega - \omega_0]/2)
And S' = \sin([\omega - (2\pi - omega_0)]*N/2)/\sin([\omega - (2\pi - \omega_0)]/2)
"""

### b
A = 1
N = 32
omega_0 = np.pi/4

omega = np.linspace(-np.pi, np.pi, int(1e7), endpoint=True)

figb, axsb = plt.subplots(ncols=2)

real_term = gen_real_term(omega, omega_0, N, A)
imag_term = gen_imag_term(omega, omega_0, N, A)

axsb[0].set_title('real term')
axsb[1].set_title('imaginary term')
axsb[0].plot(omega, real_term, linewidth=0.5)
axsb[1].plot(omega, imag_term, linewidth=0.5)

### c
n = np.arange(0, N)
x_n = A*np.cos(omega_0 * n)
X_f = scipy.fft.fftshift(scipy.fft.fft(x_n))

axsb[0].stem(n*2*np.pi/(N) - np.pi, X_f.real)
axsb[1].stem(n*2*np.pi/(N) - np.pi, X_f.imag)

### d
omega_0_new = 1.1*np.pi/4

real_term_new = gen_real_term(omega, omega_0_new, N, A)
imag_term_new = gen_imag_term(omega, omega_0_new, N, A)

axsb[0].plot(omega, real_term_new, linewidth=0.5, c="red")
axsb[1].plot(omega, imag_term_new, linewidth=0.5, c="red")

x_n_new = A*np.cos(omega_0_new * n)
X_f_new = scipy.fft.fftshift(scipy.fft.fft(x_n_new))

axsb[0].stem(n*2*np.pi/(N) - np.pi, X_f_new.real, linefmt="red")
axsb[1].stem(n*2*np.pi/(N) - np.pi, X_f_new.imag, linefmt="red")
plt.show()

"""
The DTFTs have only changed as expected. The DFTs have a more stark difference. In the original case, 32 samples constituted 4 periods of the single sinusoid. In the new case it constitutes 4.4 periods, and the introduction of that additional fraction of a period leads to artifacts in the periodic extension we assume of the DFT. While the DFT continues to be a sampled form of its DTFT, the DFT appears different due to the introduction of additional frequency content at the discontinuities of the periodic extension. While the "assumed periodicity" of the DFT can be a misleading conclusion, I think it's sound to say that there is a periodic extension happening in the IDFT considering that the full-length signal is reconstructed periodically from the finite Fourier coefficients, which creates a periodic signal as a sum of finitely many complex exponentials. One way I heuristically make that connection is that, since the DFT samples in the frequency domain, it "projects" periodicity in the time domain, just as sampling in the time domain "projects" periodicity in the frequency domain.
"""


