import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- Visual Style ---
c_Exc = [0.2, 0.6, 0.8]  # Teal
c_Inh = [0.8, 0.3, 0.3]  # Red
c_Target = [0.5, 0.5, 0.5]  # Grey
c_Input = [0.9, 0.6, 0.2]  # Orange
c_Ax = [0.15, 0.15, 0.15]  # Soft Black

# --- Simulation Parameters ---
T_sim = 2000
dt = 0.5
time = np.arange(0, T_sim + dt, dt)

# Wilson-Cowan Parameters
tau = 10
c_ee = 16
c_ei = 12
c_ii = 3
c_ie = np.zeros_like(time)
c_ie[0] = 10

# ISP Parameters
rho = 0.15  # Target Rate
eta = 0.1  # Learning Rate

# Input Perturbation
P = np.ones_like(time) * 1.5
P[time > 800] = 4.0  # Strong Step Input

# State Variables
E = np.zeros_like(time)
E[0] = rho
I = np.zeros_like(time)
I[0] = 0.1


# Sigmoid Function
def S(x):
    return 1 / (1 + np.exp(-x))


# --- Integration ---
for t in range(1, len(time)):
    # 1. Neural Dynamics
    dE = (-E[t - 1] + S(c_ee * E[t - 1] - c_ie[t - 1] * I[t - 1] + P[t - 1])) / tau
    dI = (-I[t - 1] + S(c_ei * E[t - 1] - c_ii * I[t - 1])) / tau

    E[t] = E[t - 1] + dE * dt
    I[t] = I[t - 1] + dI * dt

    # 2. Local Plasticity Rule
    # Note: I(t) gates the plasticity (Hebbian-like local rule)
    dc_ie = eta * I[t - 1] * (E[t - 1] - rho)
    c_ie[t] = c_ie[t - 1] + dc_ie * dt

    # Safety: Prevent negative weights (Biophysical constraint)
    if c_ie[t] < 0:
        c_ie[t] = 0

# --- PLOTTING ---
fig = plt.figure(figsize=(5, 6), facecolor='w')


# Helper function to apply axes styling (equivalent to MATLAB's `set(gca, ...)`)
def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(c_Ax)
    ax.spines['left'].set_color(c_Ax)
    ax.tick_params(axis='x', colors=c_Ax)
    ax.tick_params(axis='y', colors=c_Ax)


# Panel A: Excitatory Rate
ax1 = plt.subplot(3, 1, 1)
ax1.axhline(rho, linestyle='--', color=c_Target, linewidth=1.5)
ax1.plot(time, E, color=c_Exc, linewidth=2)
ax1.text(100, rho + 0.1, r'Target $\rho$', color=c_Target, fontsize=10)
ax1.set_ylabel('Firing Rate', fontsize=11, color=c_Ax)
ax1.set_xlim([0, T_sim])
ax1.set_ylim([0, 1.1])
ax1.set_xticks([])
style_axes(ax1)

# Panel B: Inhibitory Weight
ax2 = plt.subplot(3, 1, 2)
ax2.plot(time, c_ie, color=c_Inh, linewidth=2.5)
ax2.set_ylabel(r'Inhibitory Weight $c_{ie}(t)$', fontsize=11, color=c_Ax)
ax2.set_xlim([0, T_sim])
ax2.set_xticks([])
style_axes(ax2)

# Panel C: Input
ax3 = plt.subplot(3, 1, 3)
ax3.fill_between(time, P, color=c_Input, alpha=0.2, edgecolor='none')
ax3.plot(time, P, color=c_Input, linewidth=1.5)
ax3.set_xlabel('Simulation Time', fontsize=11, color=c_Ax)
ax3.set_ylabel('Input Perturbation', fontsize=11, color=c_Ax)
ax3.set_xlim([0, T_sim])
ax3.set_ylim([0, 5])
style_axes(ax3)

plt.tight_layout()
plt.show()