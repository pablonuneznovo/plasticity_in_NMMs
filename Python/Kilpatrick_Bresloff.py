import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# --- 1. Visual Style ---
c_U      = [0.2, 0.6, 0.8]    # Teal (Used for Activity U)
c_Q      = [0.8, 0.3, 0.3]    # Brick Red (Used for Resources Q)
c_Thresh = [0.5, 0.5, 0.5]    # Grey (Threshold theta)
c_Ax     = [0.15, 0.15, 0.15] # Soft Black
c_Fill   = [0.2, 0.6, 0.8]    # Teal fill for activity

# --- 2. Parameters (Kilpatrick & Bressloff 2010) ---
# Parameters from Fig 1 caption
alpha = 20        # Recovery time constant
beta  = 0.01      # Depression strength
theta = 0.2       # Firing threshold

# --- 3. Functions ---
# Weight integral function W(x) for Mexican Hat
# w(x) = (1-|x|)exp(-|x|) -> W(x) = x*exp(-|x|)
def W_func(x):
    return x * np.exp(-np.abs(x))

# Synaptic Drive U(x)
# U(x) = 1/(1+alpha*beta) * [W(x+a) - W(x-a)]
def U_func(x, a, alp, bet):
    return (1 / (1 + alp * bet)) * (W_func(x + a) - W_func(x - a))

# Depression Variable Q(x)
# Q(x) = 1 - (alpha*beta)/(1+alpha*beta) * Heaviside(U(x) - theta)
def Q_func(u_val, alp, bet, th):
    # Using > returns boolean, multiply by 1 or rely on Python casting to float
    return 1 - ((alp * bet) / (1 + alp * bet)) * (u_val > th)

# --- 4. Solve and Calculate ---
# Solve for bump half-width 'a' using threshold condition U(a) = theta
# Implicit equation: 2*a*exp(-2*a)/(1+alpha*beta) - theta = 0
def target_eq(a):
    return (2 * a * np.exp(-2 * a)) / (1 + alpha * beta) - theta

# Initial guess ~1.0
a_sol = fsolve(target_eq, 1.0)[0]

# Generate spatial domain
x = np.linspace(-6, 6, 1200) # Slightly wider X range for context

# Compute Profiles
U_vals = U_func(x, a_sol, alpha, beta)
Q_vals = Q_func(U_vals, alpha, beta, theta)

# --- 5. Plotting ---
fig, ax = plt.subplots(figsize=(7, 5.5), facecolor='w')

# === Plot Elements ===

# 1. Active Region Shading (Under U(x))
# Identify region where U > theta (Calculated but not plotted, matching original MATLAB script)
active_region = U_vals > theta

# 2. Threshold Line
ax.axhline(theta, linestyle='--', color=c_Thresh, linewidth=1.5)
ax.text(-5.5, theta + 0.06, r'$\theta$', color=c_Thresh, fontsize=16, fontweight='bold')

# 3. Synaptic Drive U(x) (Teal)
ax.plot(x, U_vals, '-', color=c_U, linewidth=3)

# 4. Depression Q(x) (Red, Dashed)
ax.plot(x, Q_vals, '--', color=c_Q, linewidth=2.5)

# === Annotations and Styling ===

# Axis Limits - EXPANDED Y-AXIS
ax.set_xlim([-6, 6])
ax.set_ylim([-0.4, 1.3])

# Labels
ax.set_xlabel('x', color=c_Ax, fontsize=16)

# Custom Text Labels (Repositioned for new limits)
ax.text(0, 0.7, 'Synaptic Drive U(x)', color=c_U, fontsize=16, fontweight='bold',
        horizontalalignment='center', bbox=dict(facecolor='w', edgecolor='none', pad=1.0))

ax.text(0.5, 1.075, 'Available Resources Q(x)', color=c_Q, fontsize=16, fontweight='bold',
        horizontalalignment='left')

# Axis Properties
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_color(c_Ax)
ax.spines['left'].set_color(c_Ax)
ax.tick_params(axis='x', colors=c_Ax, labelsize=16)
ax.tick_params(axis='y', colors=c_Ax, labelsize=16)
ax.set_yticks([-0.2, 0, theta, 0.5, 1])

plt.tight_layout()
plt.show()