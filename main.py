# Stock Price Modelling. Stochastic Differential Equations
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

plt.rcParams['figure.figsize'] = (9,6)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
pal = ["#FBB4AE","#B3CDE3", "#CCEBC5","#CFCCC4"]

# White Noise
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

# Numerical Schemes
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
        xi_array[i] = xi_array[i-1] + (1.0/alfa) * (sigma_array[i-1] - xi_array[i-1]) * delta_t
        sigma_array[i] = sigma_array[i-1] - (sigma_array[i-1] - xi_array[i-1]) * delta_t + p * sigma_array[i-1] * (Wiener2[i] - Wiener2[i-1]) + 0.5 * p**2 * sigma_array[i-1] * ((Wiener2[i] - Wiener2[i-1])**2 - delta_t)
        S_array[i] = S_array[i-1] + coeff * S_array[i-1] * delta_t + sigma_array[i-1] * S_array[i-1] * (Wiener1[i] - Wiener1[i-1]) + 0.5 * sigma_array[i-1]**2 * S_array[i-1] * ((Wiener1[i] - Wiener1[i-1])**2 - delta_t)

    return xi_array, sigma_array, S_array
def Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, timestamps, N):
    solution = np.zeros(N)
    for i, t in enumerate(timestamps):
        solution[i] = S_0 * np.exp((coeff - 0.5 * sigma_0**2) * t + Wiener1[i] * sigma_0)
    return solution

# Plotting
def plot_schemes(Euler_array, Milstein_array, alfa, p):
    plt.plot(timestamps, Euler_array, label="Euler ($S_t$)", color=pal[0])
    plt.plot(timestamps, Milstein_array, label="Milstein ($S_t$)", color=pal[1])
    # plt.plot(timestamps, B_S, label="Exact ($S_t$)", color=pal[2])
    plt.title(r"Euler vs. Milstein schemes for $p =$ {p} and $\alpha =$ {alfa}".format(p=p, alfa=alfa))
    plt.xlabel('t')
    plt.legend(loc = 2)
    plt.savefig("Plots/p_{p}_alpha_{alpha}.jpg".format(p=p, alpha=alfa))
    plt.show()

# Convergence Testing
def strong_convergence(sample_sizes, sim_size):
    # Finest grid
    N = 2 * max(sample_sizes)

    errors = []
    for n in sample_sizes:
        error_dt = strong_error(sim_size, n, N)
        errors.append((n, error_dt))
    return errors
def strong_error(sim_size, N, N_benchmark):
    Euler_error = 0.0
    Milstein_error = 0.0
    for i in range(sim_size):
        seeds = [i, i+1]
        benchmark_solution_EoI = numerical_solution(N_benchmark, seeds, scheme_name="Milstein")[N_benchmark - 1]
        Num_solution_Euler = numerical_solution(N, seeds, "Euler")
        Num_solution_Milstein = numerical_solution(N, seeds, "Milstein")
        Euler_EoI = Num_solution_Euler[N - 1]
        Milstein_EoI = Num_solution_Milstein[N - 1]
        single_error_Euler = abs(benchmark_solution_EoI - Euler_EoI)
        single_error_Milstein = abs(benchmark_solution_EoI - Milstein_EoI)
        Euler_error += single_error_Euler
        Milstein_error += single_error_Milstein
    Euler_error = Euler_error / sim_size
    Milstein_error = Milstein_error / sim_size
    return Euler_error, Milstein_error

def weak_convergence(sample_sizes, sim_size):
    # Finest grid
    N = 2 * max(sample_sizes)

    errors = []
    for n in sample_sizes:
        error_dt = weak_error(sim_size, n, N)
        errors.append((n, error_dt))
    return errors
def weak_error(sim_size, N, N_benchmark):
    Euler_error = 0.0
    Milstein_error = 0.0
    avg_Euler = 0.0
    avg_Milstein = 0.0
    avg_Benchmark = 0.0
    for i in range(sim_size):
        seeds = [i, i + 1]
        benchmark_solution_EoI = numerical_solution(N_benchmark, seeds, scheme_name="Milstein")[N_benchmark - 1]
        Num_solution_Euler = numerical_solution(N, seeds, "Euler")
        Num_solution_Milstein = numerical_solution(N, seeds, "Milstein")
        Euler_EoI = Num_solution_Euler[N - 1]
        Milstein_EoI = Num_solution_Milstein[N - 1]
        avg_Euler += Euler_EoI
        avg_Milstein += Milstein_EoI
        avg_Benchmark += benchmark_solution_EoI
    avg_Euler = avg_Euler / sim_size
    avg_Milstein = avg_Milstein / sim_size
    avg_Benchmark = avg_Benchmark / sim_size
    Euler_error = abs(avg_Benchmark - avg_Euler)
    Milstein_error = abs(avg_Benchmark - avg_Milstein)
    return Euler_error, Milstein_error

def numerical_solution(N, seeds, scheme_name):
    # Space Parameters
    T_end = 1.0
    delta_t = T_end / N
    # General Equation Parameters
    S_0 = 50.0
    sigma_0 = 0.20
    xi_0 = 0.20
    coeff = 0.1
    p = 0
    alfa = 1
    # Wiener processes
    timestamps = np.linspace(0.0, T_end, N)
    increments1 = normal_distribution(0, delta_t, N - 1, seeds[0])
    increments2 = normal_distribution(0, delta_t, N - 1, seeds[1])
    Wiener1 = Wiener_process(N, increments1)
    Wiener2 = Wiener_process(N, increments2)

    if (scheme_name == "Milstein"):
        return Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)[2]
    return Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)[2]

def plot_errors(variable, SE, WE, SM, WM):
    plt.plot(variable, SE, label="Euler Strong Error", color="#AC3015")
    plt.plot(variable, WE, label="Euler Weak Error", color="#AC3015", ls='--')
    plt.plot(variable, SM, label="Milstein Strong Error", color="#1E97DE")
    plt.plot(variable, WM, label="Milstein Weak Error", color="#1E97DE", ls='--')
    plt.xlabel('$N$');
    plt.ylabel('Error$_N$');
    plt.legend()
    plt.show()

sample_sizes = [5, 10, 20, 50, 100, 250, 500]
sim_size = 20
Strong_errors = list(zip(*list(zip(*strong_convergence(sample_sizes, sim_size)))[1]))
Weak_errors = list(zip(*list(zip(*weak_convergence(sample_sizes, sim_size)))[1]))
Strong_Euler_errors = Strong_errors[0]
Strong_Milstein_errors = Strong_errors[1]
Weak_Euler_errors = Weak_errors[0]
Weak_Milstein_errors = Weak_errors[1]

# Plotting
plot_errors(sample_sizes, Strong_Euler_errors, Weak_Euler_errors, Strong_Milstein_errors, Weak_Milstein_errors)
sys.exit() ###############################################################################

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
p_array = [0.0]
alfa_array = []

for i in range(-3, 5):
    p_array.append(10**i)
    alfa_array.append(10**i)

p = 0
alfa = 1

# Wiener processes
timestamps = np.linspace(0.0, T_end, N)
increments1 = normal_distribution(0, delta_t, N-1, 1)
increments2 = normal_distribution(0, delta_t, N-1, 2)
Wiener1 = Wiener_process(N, increments1)
Wiener2 = Wiener_process(N, increments2)

# Exact solution (Black-Scholes)
# B_S = Black_Scholes_exact(sigma_0, coeff, S_0, Wiener1, timestamps, N)

# Cartesian product of the two lists
parameter_combinations = itertools.product(alfa_array, p_array)

for pair in parameter_combinations:
    # Numerical solutions
    alfa = pair[0]
    p = pair[1]
    _, _, S_array_E = Euler_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)
    _, _, S_array_M = Milstein_scheme(xi_0, sigma_0, S_0, coeff, alfa, p, Wiener1, Wiener2, delta_t, N)
    plot_schemes(S_array_E, S_array_M, alfa, p)
