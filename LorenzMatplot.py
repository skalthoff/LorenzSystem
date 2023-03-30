import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def plot_lorenz_system(sigma, rho, beta, initial_state, t_span, t_eval):
    sol = solve_ivp(lorenz_system, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sol.y[0], sol.y[1], sol.y[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Lorenz System")
    plt.show()

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8/3

# Initial state [x, y, z]
initial_state = [0, 1, 1.05]

# Time span for the solution
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], num=10000)

plot_lorenz_system(sigma, rho, beta, initial_state, t_span, t_eval)
