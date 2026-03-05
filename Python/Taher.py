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

# %% 1. Model Parameters (Fig 1)
P = {
    # Time constants are converted to seconds
    'tau_m': 0.015,  # Membrane time constant (15 ms)
    'tau_d': 0.200,  # Depression time constant (200 ms)
    'tau_f': 1.500,  # Facilitation time constant (1500 ms)

    # Neural Parameters
    'Delta': 0.25,  # HWHM of Lorentzian excitability
    'H': 0.0,  # Median excitability
    'J': 15.0,  # Synaptic weight strength
    'I_B': -1.0,  # Background current

    # STP Parameters
    'U0': 0.2,  # Baseline utilization factor

    # Stimulus Setup
    'I_stim_amp': 2.0,
    't1_start': 0.2, 't1_end': 0.35,  # First Pulse
    't2_start': 0.5, 't2_end': 0.65  # Second Pulse
}

# %% 2. Simulation Setup
tspan = [0, 1.0]

# Initial conditions: [r; v; x; u]
y0 = [0.1, -2.0, 1.0, P['U0']]


# Exact Neural Mass ODE Function
def taher_mass_ode(t, y, P):
    r, v, x, u = y

    # External Stimulus
    I_S = 0
    if (P['t1_start'] <= t <= P['t1_end']) or (P['t2_start'] <= t <= P['t2_end']):
        I_S = P['I_stim_amp']

    # --- Exact Neural Mass Equations ---
    # 1. Firing Rate Dynamics
    drdt = (P['Delta'] / (np.pi * P['tau_m']) + 2 * r * v) / P['tau_m']

    # 2. Mean Membrane Potential Dynamics
    synaptic_input = P['J'] * u * x * r
    dvdt = (v ** 2 + P['H'] + P['I_B'] + I_S - (np.pi * P['tau_m'] * r) ** 2 + P['tau_m'] * synaptic_input) / P['tau_m']

    # 3. Depression Dynamics
    dxdt = (1 - x) / P['tau_d'] - u * x * r

    # 4. Facilitation Dynamics
    dudt = (P['U0'] - u) / P['tau_f'] + P['U0'] * (1 - u) * r

    return [drdt, dvdt, dxdt, dudt]


# Run integration
# Using max_step to ensure the solver doesn't step over the sharp stimulus edges
sol = solve_ivp(taher_mass_ode, tspan, y0, args=(P,), method='RK45',
                rtol=1e-8, atol=1e-10, t_eval=np.linspace(0, 1.0, 1000), max_step=0.005)

t = sol.t
r = sol.y[0]
v = sol.y[1]
x = sol.y[2]
u = sol.y[3]

# Reconstruct Stimulus Vector for Plotting
I_S = np.zeros_like(t)
mask_t1 = (t >= P['t1_start']) & (t <= P['t1_end'])
mask_t2 = (t >= P['t2_start']) & (t <= P['t2_end'])
I_S[mask_t1 | mask_t2] = P['I_stim_amp']

# %% 3. Visualization
fig = plt.figure(figsize=(7, 9), facecolor='w')


# Helper function to apply axes styling
def format_axis(ax, hide_xticks=False):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    if hide_xticks:
        ax.set_xticks([])
    else:
        ax.tick_params(axis='x', labelsize=12)


# --- Panel A: Firing Rate & Input Pulse (Dual Axis) ---
ax1_left = plt.subplot(3, 1, 1)
ax1_right = ax1_left.twinx()

# Right Axis: Input Stimulus (Background Area)
ax1_right.fill_between(t, I_S, facecolor=c_Input, alpha=0.2, edgecolor='none')
ax1_right.set_ylabel('Input Current', fontsize=12, color=c_Input)
ax1_right.set_ylim([0, P['I_stim_amp'] * 1.5])
ax1_right.tick_params(axis='y', colors=c_Input, labelsize=12)
ax1_right.spines['right'].set_color(c_Input)
ax1_right.spines['right'].set_linewidth(1.5)
ax1_right.spines['top'].set_visible(False)

# Left Axis: Firing Rate (Foreground Line)
ax1_left.plot(t, r, color=c_Rate, linewidth=2.0)
ax1_left.set_ylabel('Firing Rate', fontsize=12, color=c_Rate)
ax1_left.set_ylim([0, np.max(r) * 1.2])
ax1_left.set_xlim([0, 1.0])
ax1_left.tick_params(axis='y', colors=c_Rate, labelsize=12)
ax1_left.spines['left'].set_color(c_Rate)
format_axis(ax1_left, hide_xticks=True)

# Ensure the left axis line plots on top of the right axis background
ax1_left.set_zorder(ax1_right.get_zorder() + 1)
ax1_left.patch.set_visible(False)

# --- Panel B: Mean Membrane Potential ---
ax2 = plt.subplot(3, 1, 2)
ax2.plot(t, v, color=c_Volt, linewidth=1.5)
ax2.set_ylabel('Membrane potential', fontsize=12)
ax2.set_ylim([np.min(v) - 1, np.max(v) + 1])
ax2.set_xlim([0, 1.0])
ax2.tick_params(axis='y', labelsize=12)
ax2.spines['left'].set_color(c_Ax)
ax2.spines['right'].set_visible(False)
format_axis(ax2, hide_xticks=True)

# --- Panel C: Combined Synaptic Dynamics (Dual Axis) ---
ax3_left = plt.subplot(3, 1, 3)
ax3_right = ax3_left.twinx()

# 1. Effective Synaptic Gating (Background Shade)
eff_weight = u * x
ax3_left.fill_between(t, eff_weight, facecolor=c_Memory, alpha=0.15, edgecolor='none')

# 2. Left Axis: Resources (x) - Blue
ax3_left.plot(t, x, color=c_Depress, linewidth=2.0)
ax3_left.set_ylabel('Resources (q)', fontsize=12, color=c_Depress)
ax3_left.set_ylim([0, 1.1])
ax3_left.set_xlim([0, 1.0])
ax3_left.tick_params(axis='y', colors=c_Depress, labelsize=12)
ax3_left.spines['left'].set_color(c_Depress)
format_axis(ax3_left, hide_xticks=False)

# Ensure left axis is above the background
ax3_left.set_zorder(ax3_right.get_zorder() + 1)
ax3_left.patch.set_visible(False)

# 3. Right Axis: Utilization (u) - Red
ax3_right.plot(t, u, color=c_Facil, linewidth=2.0)
ax3_right.set_ylabel('Utilization (u)', fontsize=12, color=c_Facil)
ax3_right.set_ylim([0, 1.0])
ax3_right.tick_params(axis='y', colors=c_Facil, labelsize=12)
ax3_right.spines['right'].set_color(c_Facil)
ax3_right.spines['right'].set_linewidth(1.5)
ax3_right.spines['top'].set_visible(False)

ax3_left.set_xlabel('Time (s)', fontsize=12)

plt.tight_layout()
plt.show()