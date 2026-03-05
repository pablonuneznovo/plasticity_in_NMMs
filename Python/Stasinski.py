import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

# --- 1. Load Connectome ---
try:
    mat_data = sio.loadmat('SC_sample.mat')
    C = mat_data['C'].astype(float)
    np.fill_diagonal(C, 0)
    C = C / np.max(C)
except Exception:
    # Fallback if file missing
    N = 90
    C = np.random.rand(N, N)
    np.fill_diagonal(C, 0)
    C = C / np.max(C)

N = C.shape[0]

# --- 2. Jansen-Rit Parameters (Table 1) ---
A = 3.25  # Max amplitude EPSP (mV)
B = 22.0  # Max amplitude IPSP (mV)
a = 0.1  # Inv. time constant excitatory (1/ms)
b = 0.05  # Inv. time constant inhibitory (1/ms)

C1 = 135
C2 = 108  # 0.8 * C1
C3 = 33.75  # 0.25 * C1
C4 = 33.75  # 0.25 * C1

v_max = 0.0025  # Max firing rate (1/ms)
v_0 = 6.0  # Firing threshold (mV)
r = 0.56  # Steepness (1/mV)

# Global Parameters
G = 21.0  # Optimal range from paper
I_mean = 0.11  # Sub-bistable boundary input

# --- 3. dFIC Control Parameters ---
eta = 0.005  # Learning rate
tau_d = 1000  # Detector time constant (ms)
Target = 0.01  # Target y0 (mV)

# --- 4. Simulation Setup ---
dt = 0.5
T_total = 12000  # 12 Seconds (Zoomed in)
Steps = int(T_total / dt)

# State Variables (N x 6)
y = np.random.rand(N, 6) * 0.01

# dFIC Variables
wFIC = np.ones(N) * 1.0  # Start at 1.0
y0_d = np.zeros(N)
y2_d = np.zeros(N)

# History Arrays
num_saved_steps = int(Steps / 50)
Hist_y0 = np.zeros((N, num_saved_steps))
Hist_wFIC = np.zeros((N, num_saved_steps))
Hist_Time = np.zeros(num_saved_steps)
idx_plot = 0

# Identify Hub and Leaf for visualization
Deg = np.sum(C, axis=1)
idx_Hub = np.argmax(Deg)
idx_Leaf = np.argmin(Deg)

print(f'Running Stasinski dFIC (G={G:.1f}, Target={Target:.2f} mV)...')


# Sigmoid Helper Function
def sig(v):
    return 2 * v_max / (1 + np.exp(r * (v_0 - v)))


# --- 5. Simulation Loop ---
for t in range(Steps):

    # Input Calculation
    sig_y1_y2 = sig(y[:, 1] - y[:, 2])
    Network_Input = G * (C @ sig_y1_y2)
    Total_Input = I_mean + Network_Input

    # Current State
    y0 = y[:, 0];
    y1 = y[:, 1];
    y2 = y[:, 2]
    y3 = y[:, 3];
    y4 = y[:, 4];
    y5 = y[:, 5]

    # Derivatives (Jansen-Rit)
    # Eq 1B modified: S[y1 - wFIC * y2]
    dy3 = A * a * sig(y1 - wFIC * y2) - 2 * a * y3 - a ** 2 * y0
    dy4 = A * a * (Total_Input + C2 * sig(C1 * y0)) - 2 * a * y4 - a ** 2 * y1
    dy5 = B * b * (C4 * sig(C3 * y0)) - 2 * b * y5 - b ** 2 * y2

    # Integration (Euler)
    y[:, 0] = y[:, 0] + y[:, 3] * dt
    y[:, 1] = y[:, 1] + y[:, 4] * dt
    y[:, 2] = y[:, 2] + y[:, 5] * dt
    y[:, 3] = y[:, 3] + dy3 * dt
    y[:, 4] = y[:, 4] + dy4 * dt
    y[:, 5] = y[:, 5] + dy5 * dt

    # dFIC Detector Dynamics (Eq 4A, 4B)
    dy0_d = (y[:, 0] - y0_d) / tau_d
    y0_d = y0_d + dy0_d * dt

    dy2_d = (y[:, 2] - y2_d) / tau_d
    y2_d = y2_d + dy2_d * dt

    # dFIC Weight Update (Eq 4C)
    dwFIC = eta * y2_d * (y0_d - Target)
    wFIC = wFIC + dwFIC * dt

    # Store Data
    if (t + 1) % 50 == 0:
        Hist_y0[:, idx_plot] = y[:, 0]
        Hist_wFIC[:, idx_plot] = wFIC
        Hist_Time[idx_plot] = (t + 1) * dt / 1000
        idx_plot += 1

# --- 6. PLOTTING ---
fig = plt.figure(figsize=(8, 8), facecolor='w')

# Trim history (in case rounding caused an extra empty index)
Hist_Time = Hist_Time[:idx_plot]
Hist_wFIC = Hist_wFIC[:, :idx_plot]
Hist_y0 = Hist_y0[:, :idx_plot]


# Helper function for axes
def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.grid(True, linestyle='-', alpha=0.5)


# PANEL A: Whole-Network Weight Adaptation
ax1 = plt.subplot(2, 1, 1)

# 1. Plot Background (All Nodes) - Faint Grey
# Transpose Hist_wFIC so Matplotlib plots each row as a separate line
ax1.plot(Hist_Time, Hist_wFIC.T, color=[0.7, 0.7, 0.7], alpha=0.4, linewidth=0.5)

# 2. Plot Highlights (Hub/Leaf) - Thick Colors
ax1.plot(Hist_Time, Hist_wFIC[idx_Hub, :], color=[0.5, 0, 0.5], linewidth=2.5)  # Purple
ax1.plot(Hist_Time, Hist_wFIC[idx_Leaf, :], color=[0.4, 0.8, 0.4], linewidth=2.5)  # Green

# 3. Dummy Handles for Correct Legend Colors
import matplotlib.lines as mlines

h_bg = mlines.Line2D([], [], color=[0.6, 0.6, 0.6], linewidth=1, label='All Nodes')
h_hub = mlines.Line2D([], [], color=[0.5, 0, 0.5], linewidth=2.5, label='Hub Node')
h_leaf = mlines.Line2D([], [], color=[0.4, 0.8, 0.4], linewidth=2.5, label='Leaf Node')

ax1.set_ylabel('Inhibitory Weight (wFIC)', fontsize=11)
ax1.set_xlabel('Simulation Time', fontsize=11)
ax1.legend(handles=[h_bg, h_hub, h_leaf], loc='best', frameon=False)
ax1.set_xlim([0, 12])
style_axes(ax1)

# PANEL B: PSP Convergence
ax2 = plt.subplot(2, 1, 2)

# 1. Plot Background
ax2.plot(Hist_Time, Hist_y0.T, color=[0.7, 0.7, 0.7], alpha=0.4, linewidth=0.5)

# 2. Plot Highlights
ax2.plot(Hist_Time, Hist_y0[idx_Hub, :], color=[0.5, 0, 0.5], linewidth=2.0)
ax2.plot(Hist_Time, Hist_y0[idx_Leaf, :], color=[0.4, 0.8, 0.4], linewidth=2.0)

# 3. Target Line
ax2.axhline(Target, color='k', linestyle='--', linewidth=2)
# Position label dynamically near the end of the line
ax2.text(12 * 0.85, Target + 0.0005, f'Target {Target} mV', fontsize=10, verticalalignment='bottom')

ax2.set_ylabel('Pyramidal PSP (mV)', fontsize=11)
ax2.set_xlabel('Simulation Time', fontsize=11)
ax2.set_xlim([0, 12])

# Fix Y-Axis: Ensure initial transient is visible
max_val = np.max(Hist_y0)
ax2.set_ylim([0, max_val * 1.1])
style_axes(ax2)

plt.tight_layout()
plt.show()