import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Visual Style ---
c_LTP   = [0.2, 0.6, 0.8]    # Teal
c_LTD   = [0.8, 0.3, 0.3]    # Brick Red
c_Basal = [0.5, 0.5, 0.5]    # Grey
c_Net   = [0.1, 0.5, 0.3]    # Deep Emerald
c_Ax    = [0.15, 0.15, 0.15] # Soft Black

# --- 2. Parameters (Wilson et al. 2016) ---
# Thresholds: "LTD (0.15-0.5), LTP (>0.5)"
theta_d = 0.15e-6  # Depression Threshold (0.15 uM)
theta_p = 0.5e-6   # Potentiation Threshold (0.50 uM)
slope   = 0.05e-6  # Sigmoid transition width

# --- 3. Functions ---
# Helper: Sigmoid
def sigmoid(x, th, k):
    return 1 / (1 + np.exp(-(x - th) / k))

# Omega (Target Weight):
# 0.5 (Low) -> Drops to 0 (Medium/LTD) -> Rises to 1 (High/LTP)
def omega_f(c):
    return 0.5 - 0.5 * sigmoid(c, theta_d, slope) + 1.0 * sigmoid(c, theta_p, slope)

# Eta (Learning Rate):
# "Low at low levels... high at moderate to high"
# Modeled as turning on at the first threshold (theta_d)
def eta_f(c):
    return 10 * sigmoid(c, theta_d, slope)

# --- 4. Calculate Curves ---
Ca_range = np.linspace(0, 1e-6, 1000)
Omega_vals = omega_f(Ca_range)
Eta_vals   = eta_f(Ca_range)

# Drift Calculation (dW/dt)
# dW/dt = Eta * (Omega - W_current)
# W_current = 0.5 (Assuming synapse is at basal strength)
W_current = 0.5
Drift = Eta_vals * (Omega_vals - W_current)

# --- 5. Plotting ---
# Position [100 100 600 700] maps roughly to 6x7 inches
fig = plt.figure(figsize=(6, 7), facecolor='w')

# Helper function to apply axes styling
def style_axes(ax, x_color, y_color):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(x_color)
    ax.spines['left'].set_color(y_color)
    ax.tick_params(axis='x', colors=x_color)
    ax.tick_params(axis='y', colors=y_color)

# === Subplot 1: Control Functions ===
ax1 = plt.subplot(2, 1, 1)

# Patches for LTD and LTP Zones
ax1.fill_between([theta_d * 1e6, theta_p * 1e6], 0, 1.3, color=c_LTD, alpha=0.1, edgecolor='none')
ax1.fill_between([theta_p * 1e6, 1.0], 0, 1.3, color=c_LTP, alpha=0.1, edgecolor='none')

# Plot Omega (Left Axis)
ax1.plot(Ca_range * 1e6, Omega_vals, '-', color=c_Ax, linewidth=2.5)
ax1.set_ylabel(r'$\Omega$ (Target Weight)', color=c_Ax, fontsize=12)
ax1.set_yticks([0, 0.5, 1])
ax1.set_ylim([0, 1.3])
style_axes(ax1, c_Ax, c_Ax)

# Plot Eta (Right Axis)
ax2 = ax1.twinx()
ax2.plot(Ca_range * 1e6, Eta_vals, '--', color=c_Basal, linewidth=2)
ax2.set_ylabel(r'$\eta$ (Learning Rate)', color=c_Basal, fontsize=12)
ax2.set_ylim([0, 13])
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_linewidth(1.2)
ax2.spines['right'].set_color(c_Basal)
ax2.tick_params(axis='y', colors=c_Basal)

ax1.set_xlim([0, 1])
ax1.set_xlabel(r'Intracellular Calcium ($\mu$M)', color=c_Ax, fontsize=12)

# Text Annotations
# Attached to ax2 because the y-coordinates (3.15) belong to the right axis scale (0-13)
ax2.text(0.32, 3.15, 'LTD Zone', color=c_LTD, fontweight='bold', fontsize=11, horizontalalignment='center')
ax2.text(0.75, 3.15, 'LTP Zone', color=c_LTP, fontweight='bold', fontsize=11, horizontalalignment='center')


# === Subplot 2: Plasticity vs Calcium ===
ax3 = plt.subplot(2, 1, 2)

# Zero Line
ax3.axhline(0, linestyle='-', color=c_Ax, linewidth=1)

# Fill Areas
mask_ltd = Drift < 0
mask_ltp = Drift > 0
ax3.fill_between(Ca_range[mask_ltd] * 1e6, Drift[mask_ltd], 0, facecolor=c_LTD, alpha=0.3, edgecolor='none')
ax3.fill_between(Ca_range[mask_ltp] * 1e6, Drift[mask_ltp], 0, facecolor=c_LTP, alpha=0.3, edgecolor='none')

# Plot Drift Curve
ax3.plot(Ca_range * 1e6, Drift, color=c_Net, linewidth=3)

ax3.set_ylabel('Weight Change (ds/dt)', color=c_Net, fontsize=12)
ax3.set_xlabel(r'Intracellular Calcium ($\mu$M)', color=c_Ax, fontsize=12)
ax3.set_xlim([0, 1])
ax3.set_ylim([np.min(Drift) * 1.2, np.max(Drift) * 1.2])

style_axes(ax3, c_Ax, c_Net)

ax3.text(0.32, np.min(Drift) / 2, 'Depression (-)', color=c_LTD, fontsize=11, fontweight='bold', horizontalalignment='center')
ax3.text(0.75, np.max(Drift) / 2, 'Potentiation (+)', color=c_LTP, fontsize=11, fontweight='bold', horizontalalignment='center')

plt.tight_layout()
plt.show()