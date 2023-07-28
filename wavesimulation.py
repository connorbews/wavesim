from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import cmath

alpha = 0.0081
n = -128
w = [1, 0]
Lx = 1000
Ly = 1000

V = 31
g = 9.81
L = (V ** 2) / g

w_m = g / (1.026 * 31)

random_amplitudes = np.random.normal(0, 1, (abs(n) * 2, abs(n) * 2))  # Modify the size as needed
random_phases = np.random.uniform(0, 2*math.pi, (abs(n) * 2, abs(n) * 2))  # Modify the size as needed

def oceanographicspectrum(dt, spec, A):
    temp = spec.copy()

    for x in range(n, abs(n)):
        kx = 2 * math.pi * x / Lx
        for y in range(n, abs(n)):
            ky = 2 * math.pi * y / Ly

            h0 = spectrumHeight(kx, ky, x, y, A)
            h1 = spectrumHeight(-kx, -ky, x, y, A)

            omega = math.sqrt(g * math.sqrt(kx ** 2 + ky ** 2))

            if abs(omega) < 1e-6:  # Set a small threshold for omega near zero
                spectrum = 0
            else:
                theta_p = math.pi / 4
                # Evaluate the directional spectrum at the random θp
                directional_spectrum = (alpha * g**2) / (omega**5) * math.exp(-(5/4) * (w_m / omega)**4) * (math.cos(theta_p - math.atan2(ky, kx)))**2

                spectrum = (h0 + h1) * 0.5 * cmath.exp(complex(0, 1) * omega * dt) * directional_spectrum

            temp[x + abs(n)][y + abs(n)] = spectrum

    return temp

def waveSpectrum(dt, spec):
    temp = spec

    for x in range(n, abs(n)):
        kx = 2 * math.pi * x / Lx
        for y in range(n, abs(n)):
            ky = 2 * math.pi * y / Ly
            aPos = spec[x + abs(n)][y + abs(n)] * cmath.exp(complex(0, 1) * dispersionWave(kx, ky) * dt)

            temp[x + abs(n)][y + abs(n)] = aPos
    
    return temp


def dispersionWave(kx, ky):
    w = math.sqrt(g * math.sqrt(kx ** 2 + ky ** 2))

    return w

def spectrumHeight(kx, ky, x, y, A):
    k = [kx, ky]
    k1 = math.sqrt(kx ** 2 + ky ** 2)

    if abs(k1) < 1e-6:  # Set a small threshold for k1 near zero
        return 0

    p = A * (math.exp(-1 / (k1 * L ** 2)) / (k1 ** 4)) * ((abs(np.dot(k, w)) ** 2))

    # Use the pre-generated random complex numbers for amplitude and phase
    r_amp = random_amplitudes[x + abs(n)][y + abs(n)]
    r_phase = random_phases[x + abs(n)][y + abs(n)]
    
    h = 1 / math.sqrt(2) * complex(r_amp * math.cos(r_phase), r_amp * math.sin(r_phase)) * math.sqrt(p)

    return h

plt.ion()

figure, ax = plt.subplots(sharex='col', sharey='row')

spectrum = np.zeros((abs(n) * 2, abs(n) * 2), dtype=complex)
spectrum = oceanographicspectrum(0, spectrum, 1)
dt = 0.1

while True:
    temp2 = ifft2(spectrum)
    
    ax.imshow(np.abs(temp2), cmap=cm.Reds)

    figure.canvas.draw()

    figure.canvas.flush_events()

    spectrum = waveSpectrum(dt, spectrum)

    dt += 0.1
