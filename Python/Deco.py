import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os

# --- 1. Load Data ---
file_path = 'SC_sample.mat'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"'{file_path}' not found. Please upload it or place it in the same directory.")

mat_data = sio.loadmat(file_path)
C = mat_data['C'].astype(float)
N = C.shape[0]

# Zero out the diagonal
np.fill_diagonal(C, 0)

if np.max(C) > 0:
    C = C / np.max(C) * 0.2

# --- 2. Parameters ---
a_E = 310
b_E = 125
d_E = 0.16
tau_S = 100
gamma = 0.641 / 1000
w_plus = 1.4
J_NMDA = 0.15
I_0 = 0.382
G = 6.0

# =========================================================================
# METHOD A: ANALYTICAL SOLUTION
# =========================================================================
print('--- Method A: Analytical ---')

Threshold = b_E / a_E
Target_Offset = -0.026
I_target = Threshold + Target_Offset

num_val = a_E * I_target - b_E
Target_Rate = num_val / (1 - np.exp(-d_E * num_val))
S_target = (tau_S * gamma * Target_Rate) / (1 + tau_S * gamma * Target_Rate)

Term_Self = w_plus * J_NMDA * S_target
# C @ vector performs matrix multiplication
Term_Net = G * J_NMDA * (C @ np.full(N, S_target))
Term_Bias = I_0

J_Analytical = (Term_Self + Term_Net + Term_Bias - I_target) / S_target
print('Analytical Calculation: Done.')

# =========================================================================
# METHOD B: ITERATIVE SOLUTION
# =========================================================================
print('\n--- Method B: Iterative ---')

J_Iter = np.ones(N)
Delta = 0.05
Max_Epochs = 25

# Find hub and leaf nodes
node_degrees = np.sum(C, axis=1)
Idx_Hub = np.argmax(node_degrees)
Idx_Leaf = np.argmin(node_degrees)

Hist_Hub = []
Hist_Leaf = []

for epoch in range(1, Max_Epochs + 1):

    # Simulation Loop (Mean Field)
    S = np.ones(N) * S_target
    Avg_Input = np.zeros(N)

    steps = 1000
    dt = 0.5
    for t in range(1, steps + 1):
        I_net = G * J_NMDA * (C @ S)
        I_E = w_plus * J_NMDA * S - J_Iter * S + I_net + I_0

        num = a_E * I_E - b_E

        # Safely compute r to avoid division by zero or negative rates
        r = np.zeros_like(num)
        mask_small = np.abs(num) < 1e-9
        r[mask_small] = 1.0 / d_E
        r[~mask_small] = num[~mask_small] / (1 - np.exp(-d_E * num[~mask_small]))
        r[r < 0] = 0

        dS = -S / tau_S + (1 - S) * gamma * r
        S = S + dS * dt

        if t > 200:
            Avg_Input += I_E

    Avg_Input = Avg_Input / (steps - 200)

    # Store History
    Hist_Hub.append(J_Iter[Idx_Hub])
    Hist_Leaf.append(J_Iter[Idx_Leaf])

    # Update Rule
    diff = Avg_Input - I_target
    J_Iter[diff > 0.001] += Delta
    J_Iter[diff < -0.001] -= Delta

    print(f'Epoch {epoch}: Hub J = {J_Iter[Idx_Hub]:.3f}')

# =========================================================================
# PLOTTING (2x1 Layout)
# =========================================================================
# Position [100 50 600 800] in MATLAB is roughly 6x8 inches in Matplotlib
fig = plt.figure(figsize=(6, 8), facecolor='w')

# --- Panel A: Convergence of J ---
ax1 = plt.subplot(2, 1, 1)

# Analytical Benchmarks (Dashed Lines)
ax1.axhline(J_Analytical[Idx_Hub], linestyle='--', color=[0.5, 0, 0.5], linewidth=2, label='Theoretical target (Hub)')
ax1.axhline(J_Analytical[Idx_Leaf], linestyle='--', color=[0.4, 0.8, 0.4], linewidth=2,
            label='Theoretical target (Leaf)')

# Iterative Trajectories (epochs are 1-indexed for plotting)
epochs_x = np.arange(1, Max_Epochs + 1)
ax1.plot(epochs_x, Hist_Hub, '-o', color=[0.5, 0, 0.5], markerfacecolor=[0.5, 0, 0.5], linewidth=1.5,
         label='Hub (Simulation)')
ax1.plot(epochs_x, Hist_Leaf, '-s', color=[0.4, 0.8, 0.4], markerfacecolor=[0.4, 0.8, 0.4], linewidth=1.5,
         label='Leaf (Simulation)')

ax1.set_ylabel('Inhibitory Weight $J_i$', fontsize=11)
ax1.set_xlabel('Optimization Epoch', fontsize=11)

# Match MATLAB's legend location and box style
ax1.legend(loc='center right', fontsize=9, frameon=False)

ax1.grid(True)
ax1.set_box_aspect(1)  # Equivalent to MATLAB's `axis square`
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.2)
ax1.spines['left'].set_linewidth(1.2)

# --- Panel B: Resulting Rates ---
ax2 = plt.subplot(2, 1, 2)

# Python plots bars centered on 0 by default, so we shift to 1-indexed to match MATLAB
nodes_x = np.arange(1, N + 1)
ax2.bar(nodes_x, r, color=[0.8, 0.3, 0.3], edgecolor='none')
ax2.axhline(Target_Rate, color='k', linestyle='--', linewidth=2, label=f'Target {Target_Rate:.2f} Hz')

# Adding the label text near the line, similarly to MATLAB's 'Label' property in yline
ax2.text(N * 0.85, Target_Rate + 0.15, f'Target {Target_Rate:.2f} Hz', fontsize=10, verticalalignment='bottom')

ax2.set_title('Final Firing Rates', fontsize=12, fontweight='bold')
ax2.set_ylabel('Firing Rate (Hz)', fontsize=11)
ax2.set_xlabel('Node Index', fontsize=11)

ax2.set_xlim([0, N + 1])
ax2.set_ylim([0, 4.5])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_linewidth(1.2)
ax2.spines['left'].set_linewidth(1.2)

plt.tight_layout()
plt.show()