import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# --- Colour Palette ---
c_Rate = [0.2, 0.2, 0.2]  # Dark Grey (r)
c_Input = [0.6, 0.6, 0.6]  # Light Grey (Input Pulse)
c_Volt = [0.3, 0.5, 0.3]  # Muted Green (v)
c_Facil = [0.8, 0.3, 0.3]  # Red (u) - Utilization
c_Depress = [0.2, 0.4, 0.8]  # Blue (x) - Resources
c_Memory = [0.6, 0.4, 0.8]  # Muted Purple (Effective Weight)
c_Ax = [0.15, 0.15, 0.15]  # Soft Black

# %% 1. Global Parameters (Fig 5/6 configuration)
P = {
    'M': 100,  # Number of subpopulations
    'tau': 1.0,  # Membrane time constant
    'Delta': 2.0,  # HWHM of global eta distribution
    'eta_bar': -3.0,  # Center of global eta distribution
    'tau_x': 50.0,  # Depression time constant
    'tau_u': 20.0,  # Facilitation time constant
    'U0': 0.2,  # Baseline synaptic efficacy
    'alpha': 0.1,  # Depression strength
    'I_amp': 3.0,  # Input pulse amplitude
    't1_start': 20.0, 't1_end': 40.0,
    't2_start': 60.0, 't2_end': 80.0
}
P['J'] = 15.0 * np.sqrt(P['Delta'])  # Global coupling strength

# 2. Subpopulation Parameterization (Eq. 22a, 22b)
m_idx = np.arange(1, P['M'] + 1)
P['eta_m'] = P['eta_bar'] + P['Delta'] * np.tan(np.pi * (2 * m_idx - P['M'] - 1) / (2 * (P['M'] + 1)))
P['Delta_m'] = P['Delta'] * (np.tan(np.pi * (2 * m_idx - P['M'] - 0.5) / (2 * (P['M'] + 1))) -
                             np.tan(np.pi * (2 * m_idx - P['M'] - 1.5) / (2 * (P['M'] + 1))))

# %% 3. Simulation Setup
tspan = [0, 100]

# Initial conditions: [r_m; v_m; x_m; u_m] x M
y0 = np.concatenate([
    np.full(P['M'], 0.2),  # r
    np.full(P['M'], -1.5),  # v
    np.full(P['M'], 0.5),  # x
    np.full(P['M'], 0.3)  # u
])


# MPA ODE Function
def gast_mpa_ode(t, y, P):
    M = P['M']
    r = y[0:M]
    v = y[M:2 * M]
    x = y[2 * M:3 * M]
    u = y[3 * M:4 * M]

    I_ext = 0
    if (P['t1_start'] <= t <= P['t1_end']) or (P['t2_start'] <= t <= P['t2_end']):
        I_ext = P['I_amp']

    # Effective network input (Weighted sum across subpopulations)
    r_eff = (P['J'] * P['tau'] / M) * np.sum(x * u * r)

    # Dynamics for each subpopulation m
    drdt = (P['Delta_m'] / (np.pi * P['tau']) + 2 * r * v) / P['tau']
    dvdt = (v ** 2 + P['eta_m'] + I_ext + r_eff - (np.pi * r * P['tau']) ** 2) / P['tau']
    dxdt = (1 - x) / P['tau_x'] - P['alpha'] * u * x * r
    dudt = (P['U0'] - u) / P['tau_u'] + P['U0'] * (1 - u) * r

    return np.concatenate([drdt, dvdt, dxdt, dudt])


# Run integration
sol = solve_ivp(gast_mpa_ode, tspan, y0, args=(P,), method='RK45',
                rtol=1e-6, atol=1e-8, t_eval=np.linspace(0, 100, 1000))

t = sol.t
y = sol.y

# Average across subpopulations for plotting
r_avg = np.mean(y[0:P['M'], :], axis=0)
v_avg = np.mean(y[P['M']:2 * P['M'], :], axis=0)
x_avg = np.mean(y[2 * P['M']:3 * P['M'], :], axis=0)
u_avg = np.mean(y[3 * P['M']:4 * P['M'], :], axis=0)

# Reconstruct Stimulus Vector for Plotting
I_S = np.zeros_like(t)
mask_t1 = (t >= P['t1_start']) & (t <= P['t1_end'])
mask_t2 = (t >= P['t2_start']) & (t <= P['t2_end'])
I_S[mask_t1 | mask_t2] = P['I_amp']

# %% 4. Visualization
fig = plt.figure(figsize=(7, 9), facecolor='w')


# Helper to remove top/bottom spines and format appropriately
def format_axis(ax, hide_xticks=False):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    if hide_xticks:
        ax.set_xticks([])
    else:
        ax.tick_params(axis='x', labelsize=14)


# --- Panel A: Firing Rate & Input Pulse ---
ax1_left = plt.subplot(3, 1, 1)
ax1_right = ax1_left.twinx()

# Right Axis: Input Stimulus (Background Area)
ax1_right.fill_between(t, I_S, facecolor=c_Input, alpha=0.2, edgecolor='none')
ax1_right.set_ylabel('Input I(t)', fontsize=16, color=c_Input)
ax1_right.set_ylim([0, P['I_amp'] * 1.5])
ax1_right.tick_params(axis='y', colors=c_Input, labelsize=14)
ax1_right.spines['right'].set_color(c_Input)
ax1_right.spines['right'].set_linewidth(1.5)
ax1_right.spines['top'].set_visible(False)

# Left Axis: Firing Rate (Foreground Line)
ax1_left.plot(t, r_avg, color=c_Rate, linewidth=2.5)
ax1_left.set_ylabel('Firing Rate r', fontsize=16, color=c_Rate)
ax1_left.set_ylim([0, 1])
ax1_left.set_xlim([0, 100])
ax1_left.tick_params(axis='y', colors=c_Rate, labelsize=14)
ax1_left.spines['left'].set_color(c_Rate)
format_axis(ax1_left, hide_xticks=True)

# --- Panel B: Membrane Potential ---
ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, v_avg, color=c_Volt, linewidth=2.0)
ax2.set_ylabel('Potential v', fontsize=16)
ax2.set_ylim([-2, 0.5])
ax2.set_xlim([0, 100])
ax2.tick_params(axis='y', labelsize=14)
ax2.spines['left'].set_color(c_Ax)
ax2.spines['right'].set_visible(False)
format_axis(ax2, hide_xticks=True)

# --- Panel C: Combined Synaptic Dynamics (Dual Axis) ---
ax3_left = plt.subplot(3, 1, 3)
ax3_right = ax3_left.twinx()

# 1. Effective Weight (Background Shade)
eff_weight = u_avg * x_avg
ax3_left.fill_between(t, eff_weight, facecolor=c_Memory, alpha=0.15, edgecolor='none')

# 2. Left Axis: Resources (x) - Blue
ax3_left.plot(t, x_avg, color=c_Depress, linewidth=2.5)
ax3_left.set_ylabel('Resources (x)', fontsize=16, color=c_Depress)
ax3_left.set_ylim([0.2, 0.8])
ax3_left.set_xlim([0, 100])
ax3_left.tick_params(axis='y', colors=c_Depress, labelsize=14)
ax3_left.spines['left'].set_color(c_Depress)
format_axis(ax3_left, hide_xticks=False)

# 3. Right Axis: Utilization (u) - Red
ax3_right.plot(t, u_avg, color=c_Facil, linewidth=2.5)
ax3_right.set_ylabel('Utilization (u)', fontsize=16, color=c_Facil)
ax3_right.set_ylim([0.2, 0.8])
ax3_right.tick_params(axis='y', colors=c_Facil, labelsize=14)
ax3_right.spines['right'].set_color(c_Facil)
ax3_right.spines['right'].set_linewidth(1.5)
ax3_right.spines['top'].set_visible(False)

ax3_left.set_xlabel('Simulation Time', fontsize=16)

plt.tight_layout()
plt.show()