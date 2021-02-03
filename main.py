# Stock Price Modelling. Stochastic Differential Equations
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
pal = ["#FBB4AE","#B3CDE3", "#CCEBC5","#CFCCC4"]

def normal_distribution(mean, var, N_samples, seed):
    np.random.seed(seed)
    return np.random.normal(mean, var, N_samples)

def Wiener_process(N, increments):
    Wiener = np.zeros(N)
    for i, element in enumerate(Wiener):
        if(i == 0):
            continue
        Wiener[i] = Wiener[i-1] + increments[i-1]
    return Wiener


def Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N):
    xi_array = np.full(N, xi_0)
    sigma_array = np.full(N, sigma_0)
    S_array = np.full(N, S_0)

    for i in range(1, N):
        xi_array[i] = xi_array[i-1] + (1.0/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - (sigma_array[i-1] - xi_array[i-1]) * delta_t + p * sigma_array[i-1] * (Wiener2[i] - Wiener2[i-1])
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener1[i] - Wiener1[i-1])

    return xi_array, sigma_array, S_array

def Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N):
    xi_array = np.full(N, xi_0)
    sigma_array = np.full(N, sigma_0)
    S_array = np.full(N, S_0)

    for i in range(1, N):
        xi_array[i] = xi_array[i-1] + (1/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - (sigma_array[i-1] - xi_array[i-1]) * delta_t + p * sigma_array[i-1] * (Wiener2[i] - Wiener2[i-1]) + 0.5 * p**2 * sigma_array[i-1] * ((Wiener2[i] - Wiener2[i-1])**2 - delta_t)
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener1[i] - Wiener1[i-1]) + 0.5 * sigma_array[i-1]**2 * S_array[i-1] * ((Wiener1[i] - Wiener1[i-1])**2 - delta_t)

    return xi_array, sigma_array, S_array

def Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, timestamps, N):
    solution = np.zeros(N)
    for i, t in enumerate(timestamps):
        solution[i] = S_0 * np.exp((coeff - 0.5 * sigma_0**2) * t + Wiener1[i] * sigma_0)
    return solution

# Space Parameters
T_end = 1.0
N = 50
delta_t = T_end/N

# General Equation Parameters
S_0 = 50.0
sigma_0 = 0.20
xi_0 = 0.20
coeff = 0.1

# Variable Parameters
p = 0
alfa = 1

# Wiener process
timestamps = np.linspace(0.0, T_end, N)
increments1 = normal_distribution(0, delta_t, N-1, 1)
increments2 = normal_distribution(0, delta_t, N-1, 2)
Wiener1 = Wiener_process(N, increments1)
Wiener2 = Wiener_process(N, increments2)

# Exact solution (Black-Scholes)
B_S = Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, timestamps, N)

# Numerical solutions
_, _, S_array_E = Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)
_, _, S_array_M = Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)

# plt.plot(timestamps, Wiener1, label="Wiener", color=pal[0])
# plt.plot(timestamps, Wiener2, label="Wiener", color=pal[1])
# plt.show()

plt.plot(timestamps, S_array_E, label="Euler ($S_t$)", color=pal[0])
plt.plot(timestamps, S_array_M, label="Milstein ($S_t$)", color=pal[1])
plt.plot(timestamps, B_S, label="Exact ($S_t$)", color=pal[2])
# plt.plot(t, X_em_small, label="EM ($X_t$): Fine Grid", color=pal[1], ls='--')
#plt.plot(coarse_grid, X_em_big, label="EM ($X_t$): Coarse Grid", color=pal[2], ls='--')
plt.title('Euler vs. Milstein schemes')
plt.xlabel('t')
plt.legend(loc = 2)
plt.show()
