import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# --- Visual Style ---
c_GDP = [0.2, 0.6, 0.8]  # Teal (Distance Rule)
c_SDP = [0.8, 0.3, 0.3]  # Red (Sync Rule)
c_Ax = [0.15, 0.15, 0.15]  # Dark Grey (Axes/Text)
colormap_style = 'gray_r'  # Equivalent to flipud(gray)

# --- Simulation Parameters ---
N = 40
T_sim = 600
dt = 0.05

# 1. Initial Topology (Random)
W = (np.random.rand(N, N) > 0.80).astype(float)
np.fill_diagonal(W, 0)
W = (W + W.T) / 2
W[W > 0] = 0.5
W_start = W.copy()

# 2. Dynamics Setup (Kuramoto)
theta = 2 * np.pi * np.random.rand(N)
omega = 1.0 + 0.1 * np.random.randn(N)
coupling_k = 2.0

# 3. Plasticity Parameters
alpha_SDP = 0.02
threshold_SDP = 0.4
Target_Deg = 4

# --- Evolution Loop ---
for t in range(1, T_sim + 1):
    # Fast Dynamics
    # Broadcasting to get phase_diff[i, j] = theta[j] - theta[i]
    phase_diff = theta[None, :] - theta[:, None]
    interaction = np.sum(W * np.sin(phase_diff), axis=1)
    theta = theta + (omega + coupling_k / N * interaction) * dt

    # Slow Dynamics
    if t % 5 == 0:
        Sync = np.cos(phase_diff)

        # A. SDP (Hebbian)
        dW = alpha_SDP * (Sync - threshold_SDP)
        mask = W > 0
        W[mask] = W[mask] + dW[mask]
        W[W < 0.05] = 0  # Pruning
        W = (W + W.T) / 2

        # B. GDP (Homeostatic)
        degrees = np.sum(W > 0, axis=1)
        for i in range(N):
            if degrees[i] < Target_Deg:
                dist_vec = np.abs(np.arange(N) - i)
                dist_vec = np.minimum(dist_vec, N - dist_vec)
                prob = np.exp(-dist_vec / 4.0)
                prob[i] = 0
                prob[W[i, :] > 0] = 0

                sum_prob = np.sum(prob)
                if sum_prob > 0:
                    prob = prob / sum_prob  # Normalize to create valid probabilities
                    target = np.random.choice(N, p=prob)
                    W[i, target] = 0.5
                    W[target, i] = 0.5

 # --- PLOTTING ---
fig = plt.figure(figsize=(6, 6), facecolor='w')


# Helper function to apply axes styling
def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_color(c_Ax)
    ax.spines['left'].set_color(c_Ax)
    ax.tick_params(axis='x', colors=c_Ax)
    ax.tick_params(axis='y', colors=c_Ax)


# Panel 1: GDP Rule (Distance)
ax1 = plt.subplot(2, 2, 1)
d = np.arange(16)
target_w = np.exp(-0.2 * d)  # Stam Eq. 3
ax1.plot(d, target_w, color=c_GDP, linewidth=2.5)
ax1.set_title('GDP Rule', fontweight='bold', color=c_Ax)
ax1.set_xlabel('Node Distance', color=c_Ax)
ax1.set_ylabel('Target Strength', color=c_Ax)
ax1.set_xlim([0, 15])
ax1.set_ylim([0, 1])
style_axes(ax1)

# Panel 2: SDP Rule (Synchronization)
ax2 = plt.subplot(2, 2, 2)
r = np.linspace(0, 2, 100)
Hill = (r ** 2) / (r ** 2 + 1) - 0.5  # Stam Eq. 4
ax2.plot(r, Hill, color=c_SDP, linewidth=2.5)
ax2.axhline(0, linestyle='-', color=[0.7, 0.7, 0.7])
ax2.axvline(1, linestyle='--', color=[0.7, 0.7, 0.7])
ax2.set_title('SDP Rule', fontweight='bold', color=c_Ax)
ax2.set_xlabel('Synchronization (r)', color=c_Ax)
ax2.set_ylabel(r'$\Delta$ Weight', color=c_Ax)
ax2.set_xlim([0, 2])
ax2.set_ylim([-0.6, 0.6])
style_axes(ax2)

# Panel 3: Start Matrix
ax3 = plt.subplot(2, 2, 3)
ax3.imshow(W_start, cmap=colormap_style, aspect='equal', interpolation='nearest')
ax3.axis('off')
ax3.set_title('Network Weights (Start)', fontweight='bold', color=c_Ax)

# Panel 4: End Matrix
ax4 = plt.subplot(2, 2, 4)
idx = np.argsort(theta)
# Rearrange the matrix rows and columns based on the sorted phases
W_sorted = W[np.ix_(idx, idx)]
ax4.imshow(W_sorted, cmap=colormap_style, aspect='equal', interpolation='nearest')
ax4.axis('off')
ax4.set_title('Network Weights (End)', fontweight='bold', color=c_Ax)

plt.tight_layout()
plt.show()