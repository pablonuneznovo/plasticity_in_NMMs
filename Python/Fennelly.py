import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- Visual Style ---
c_Target = [0.5, 0.3, 0.7]    # Purple (Target Function)
c_K      = [0.2, 0.6, 0.8]    # Teal (Coupling k)
c_Sync   = [0.8, 0.4, 0.2]    # Orange (Synchrony |z|)
c_Ax     = [0.15, 0.15, 0.15] # Soft Black

# --- Parameters ---
alpha = 2.0
epsilon = 0.1

# --- Simulation ---
T_sim = 200
dt = 0.1
time = np.arange(0, T_sim + dt, dt)

# 1. Input: Synchrony |z|
z_mag = np.zeros_like(time)
z_mag[time < 50] = 0.1
z_mag[(time >= 50) & (time < 120)] = 0.9
z_mag[time >= 120] = 0.4

# 2. Dynamics: dk/dt = epsilon * (-k + alpha * |z|^2)
k = np.zeros_like(time)
k[0] = 0.1

for i in range(1, len(time)):
    dk = epsilon * (-k[i-1] + alpha * z_mag[i-1]**2)
    k[i] = k[i-1] + dk * dt

# --- PLOTTING ---
# Position [100 100 400 600] maps roughly to 4x6 inches
fig = plt.figure(figsize=(4, 6), facecolor='w')

# Panel 1: The Rule
ax1 = plt.subplot(2, 1, 1)

r_range = np.linspace(0, 1, 100)
Target_Curve = alpha * r_range**2

ax1.plot(r_range, Target_Curve, color=c_Target, linewidth=3)
ax1.fill_between(r_range, Target_Curve, 0, facecolor=c_Target, alpha=0.1, edgecolor='none')

ax1.set_title('Macroscopic Rule', fontsize=12, fontweight='bold', color=c_Ax)
ax1.set_ylabel(r'Target Coupling $\alpha |z|^2$', fontsize=11, color=c_Ax)
ax1.set_xlabel('Synchrony |z|', fontsize=11, color=c_Ax)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, alpha * 1.1])

# Axis styling for ax1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_color(c_Ax)
ax1.spines['left'].set_color(c_Ax)
ax1.tick_params(axis='x', colors=c_Ax)
ax1.tick_params(axis='y', colors=c_Ax)

# Panel 2: Dynamics
ax2 = plt.subplot(2, 1, 2)

# Synchrony (Left Axis)
ax2.plot(time, z_mag, ':', color=c_Sync, linewidth=2)
ax2.set_ylabel('Synchrony |z|(t)', fontsize=11, fontweight='bold', color=c_Sync)
ax2.tick_params(axis='y', colors=c_Sync)
ax2.set_ylim([0, 1.1])
ax2.spines['left'].set_color(c_Sync)

# Coupling k (Right Axis - using twinx)
ax3 = ax2.twinx()
ax3.plot(time, k, color=c_K, linewidth=2.5)
ax3.set_ylabel('Mean Coupling k(t)', fontsize=12, fontweight='bold', color=c_K)
ax3.tick_params(axis='y', colors=c_K)
ax3.set_ylim([0, alpha])
ax3.spines['right'].set_color(c_K)

ax2.set_title('Adaptation Dynamics', fontsize=12, fontweight='bold', color=c_Ax)
ax2.set_xlabel('Simulation Time', fontsize=11, color=c_Ax)
ax2.set_xlim([0, T_sim])

# Axis styling for ax2/ax3
ax2.spines['top'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['left'].set_linewidth(1.2)
ax3.spines['right'].set_linewidth(1.2)
ax2.spines['bottom'].set_color(c_Ax)
ax2.tick_params(axis='x', colors=c_Ax)

plt.tight_layout()
plt.show()