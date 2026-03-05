import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- Parameters (from Milstein et al. 2021 Results) ---
tau_ET = 1000  # ms (Eligibility Trace decay)
tau_IS = 500  # ms (Instructive Signal decay)
W_max = 4.0  # Maximum weight
k_plus = 2.0  # Potentiation rate
k_minus = 0.5  # Depression rate (lower than potentiation)

# Sigmoid Parameters for q+ and q- (Eqs 10-13)
alpha_plus = 0.2
beta_plus = 30
alpha_minus = 0.1
beta_minus = 30  # Depression has lower threshold

# --- Simulation Settings ---
T = 6000
dt = 1
time = np.arange(1, T + 1, dt)

# 1. Input Spikes (Eligibility)
t_spike = 1500
spikes = np.zeros(T)
spikes[t_spike - 1] = 1  # 0-indexed adjustment

# 2. Plateau Potential (Instructive Signal)
t_plat_onset = 2500
dur_plat = 300
Plateau_Gate = np.zeros(T)
# 0-indexed adjustment for inclusive range equivalent to MATLAB's 2500:2800
Plateau_Gate[(t_plat_onset - 1): (t_plat_onset + dur_plat)] = 1

# --- Integration of Signal Dynamics ---
ET = np.zeros(T)
IS = np.zeros(T)

for t in range(1, T):
    # Eq. 8: Eligibility Trace
    dET = -ET[t - 1] / tau_ET + spikes[t - 1]
    ET[t] = ET[t - 1] + dET * dt

    # Eq. 9: Instructive Signal
    dIS = -IS[t - 1] / tau_IS + Plateau_Gate[t - 1]
    IS[t] = IS[t - 1] + dIS * dt

# Normalize IS for calculating overlap (conceptually)
IS = IS / np.max(IS)

# Calculate Signal Overlap
Overlap = ET * IS

# --- Integration of Synaptic Weight (Eq. 16) ---
# We simulate 3 synapses with different INITIAL weights to show the rule (Script simulates 2)
W_weak = np.zeros(T)
W_weak[0] = 0.5
W_strong = np.zeros(T)
W_strong[0] = 3.5

for t in range(1, T):
    # q functions (Sigmoidal dependence on Overlap)
    # Using generic sigmoid function s(x) = 1 / (1 + exp(-beta*(x-alpha)))
    q_plus = 1 / (1 + np.exp(-beta_plus * (Overlap[t - 1] - alpha_plus)))
    q_minus = 1 / (1 + np.exp(-beta_minus * (Overlap[t - 1] - alpha_minus)))

    # Eq. 16: Weight Dynamics
    # Weak Synapse
    dW_w = (W_max - W_weak[t - 1]) * k_plus * q_plus - W_weak[t - 1] * k_minus * q_minus
    W_weak[t] = W_weak[t - 1] + dW_w * dt * 0.001  # Scale time

    # Strong Synapse
    dW_s = (W_max - W_strong[t - 1]) * k_plus * q_plus - W_strong[t - 1] * k_minus * q_minus
    W_strong[t] = W_strong[t - 1] + dW_s * dt * 0.001

# --- PLOTTING ---
fig = plt.figure(figsize=(6, 8), facecolor='w')


# Helper function to apply axes styling
def style_axes(ax, hide_xticks=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.tick_params(axis='y', labelsize=16)
    if hide_xticks:
        ax.set_xticks([])
    else:
        ax.tick_params(axis='x', labelsize=16)


# Panel 1: Signals
ax1 = plt.subplot(3, 1, 1)
ax1.plot(time, ET, color=[0.2, 0.6, 0.8], linewidth=2, label='Eligibility Trace')
ax1.plot(time, IS, color=[0.8, 0.3, 0.3], linewidth=2, label='Instructive Signal')

# Fill plateau region
ax1.fill_between([t_plat_onset, t_plat_onset + dur_plat], 0, 1,
                 facecolor=[0.8, 0.3, 0.3], alpha=0.1, edgecolor='none')

ax1.legend(frameon=False, loc='upper right', fontsize=10)
ax1.set_ylabel('Amplitude', fontsize=16)
ax1.set_xlim([0, T])
style_axes(ax1, hide_xticks=True)

# Panel 2: Overlap
ax2 = plt.subplot(3, 1, 2)
ax2.fill_between(time, Overlap, 0, facecolor=[0.1, 0.5, 0.3], alpha=0.5, edgecolor='none')
ax2.set_ylabel('Overlap', fontsize=16)
ax2.set_xlim([0, T])
ax2.set_ylim([0, 0.4])
style_axes(ax2, hide_xticks=True)

# Panel 3: Weight Change
ax3 = plt.subplot(3, 1, 3)
ax3.plot(time, W_weak, color=[0.2, 0.6, 0.2], linewidth=2.5, label='Weak → Potentiates')
ax3.plot(time, W_strong, color=[0.8, 0.2, 0.2], linewidth=2.5, label='Strong → Depresses')
ax3.axhline(W_max, linestyle=':', color=[0.5, 0.5, 0.5])

# Note: MATLAB script specifically specified FontSize 10 for the xlabel here
ax3.set_xlabel('Simulation Time', fontsize=10)
ax3.set_ylabel('Synaptic Weight', fontsize=16)
ax3.legend(frameon=False, loc='center right', fontsize=10)
ax3.set_xlim([0, T])
ax3.set_ylim([0, 4.5])
style_axes(ax3, hide_xticks=False)

plt.tight_layout()
plt.show()