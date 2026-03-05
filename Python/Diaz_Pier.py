import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- Modern Palette ---
c_Ca = [0.2, 0.6, 0.8]  # Teal (Calcium)
c_Struct = [0.5, 0.3, 0.7]  # Purple (Synaptic Elements z)
c_Zone = [0.2, 0.8, 0.4]  # Green (Growth Window)
c_Ax = [0.15, 0.15, 0.15]  # Soft Black

# --- Parameters (Diaz-Pier et al., 2016) ---
# nu: Growth rate. Set to 4.0 x 10^-4 elements/ms
nu = 0.0004
# eta: Minimum activity threshold. Set to 0.0
eta = 0.0
# epsilon: Target calcium concentration. Set to 0.2 (Inhibitory target)
epsilon = 0.2

# Derived Gaussian Parameters
xi = (eta + epsilon) / 2  # Center of window
zeta = (epsilon - eta) / (2 * np.sqrt(np.log(2)))  # Width parameter

# --- Simulation Setup ---
T = 25000
time = np.arange(1, T + 1)

# 1. Calcium Sweep (0 to 0.6)
Ca = np.linspace(0, 0.6, T)

# 2. Integration of Structural Elements (z)
z = np.zeros(T)
z[0] = 0.8

for t in range(1, T):
    Current_Ca = Ca[t]

    # Exact Gaussian Equation
    growth_speed = nu * (2 * np.exp(-((Current_Ca - xi) / zeta) ** 2) - 1)

    # Update z (Forward Euler)
    z[t] = z[t - 1] + growth_speed
    z[t] = max(0.0, z[t])

# --- PLOTTING ---
# Position [100 100 400 500] maps roughly to 4x5 inches
fig = plt.figure(figsize=(4, 5), facecolor='w')

# Subplot 1: The Gaussian Growth Rule (Function of Calcium)
ax1 = plt.subplot(2, 1, 1)

# Theoretical Curve for visualization
Ca_range = np.linspace(0, 0.6, 200)
Growth_Curve = nu * (2 * np.exp(-((Ca_range - xi) / zeta) ** 2) - 1)

# Zero Line
ax1.axhline(0, color=c_Ax, linewidth=1)

# Fill Growth Zone (Positive Growth)
mask = Growth_Curve > 0
ax1.fill_between(Ca_range[mask], Growth_Curve[mask], 0,
                 facecolor=c_Zone, alpha=0.2, edgecolor='none')

# Plot Curve
ax1.plot(Ca_range, Growth_Curve, 'k', linewidth=2)

# Annotations
ax1.axvline(eta, linestyle='--', color=c_Zone, linewidth=1.5)
ax1.axvline(epsilon, linestyle='--', color=[0.8, 0.2, 0.2], linewidth=1.5)

# Text Offset
ax1.text(epsilon + 0.02, nu * 1.1, r'$\epsilon$ (Target)', color=[0.8, 0.2, 0.2],
         fontsize=9, horizontalalignment='left')

ax1.set_ylabel('Growth Rate dz/dt', fontsize=10, fontweight='bold', color=c_Ax)
ax1.set_xlabel('Calcium concentration', fontsize=10, fontweight='bold', color=c_Ax)
ax1.set_title('Gaussian Growth Rule', fontsize=12, fontweight='bold', color=c_Ax)
ax1.set_xlim([0, 0.6])
ax1.set_ylim([-nu, nu * 1.3])
ax1.set_yticks([])

# Axis styling for ax1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_color(c_Ax)
ax1.spines['left'].set_color(c_Ax)
ax1.tick_params(axis='x', colors=c_Ax)

# Subplot 2: Temporal Evolution (Result)
ax2 = plt.subplot(2, 1, 2)

# Visualize the Calcium Input (Background) - Left Axis
ax2.plot(time, Ca, color=c_Ca, linewidth=2, linestyle=':')
ax2.set_ylabel(r'Intracellular $Ca^{2+}$', fontsize=10, color=c_Ca)
ax2.set_ylim([0, 0.6])
ax2.tick_params(axis='y', colors=c_Ca)
ax2.spines['left'].set_color(c_Ca)

# Visualize the Synaptic Elements (Foreground) - Right Axis (twinx)
ax3 = ax2.twinx()
ax3.plot(time, z, color=c_Struct, linewidth=2.5)
ax3.set_ylabel('Synaptic Elements z(t)', fontsize=10, fontweight='bold', color=c_Struct)
ax3.tick_params(axis='y', colors=c_Struct)
ax3.spines['right'].set_color(c_Struct)
ax3.spines['right'].set_linewidth(1.2)

# Auto-scale Y-axis for ax3
ymax = np.max(z)
if ymax == 0:
    ymax = 1
ax3.set_ylim([0, ymax * 1.1])

ax2.set_xlabel('Simulation Time', fontsize=10, fontweight='bold', color=c_Ax)
ax2.set_xlim([0, T])
ax2.grid(True, linestyle='-', alpha=0.5)

# Axis styling for ax2/ax3
ax2.spines['top'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['left'].set_linewidth(1.2)
ax2.spines['bottom'].set_color(c_Ax)
ax2.tick_params(axis='x', colors=c_Ax)

plt.tight_layout()
plt.show()