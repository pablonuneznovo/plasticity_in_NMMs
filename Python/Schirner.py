import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from scipy.interpolate import Akima1DInterpolator


# =========================================================================
# CORE SIMULATION
# =========================================================================
def simulate_trial(P, BW, w_LRE, w_FFI, G, sigma):
    # Steady State FIC (Analytic)
    r_targ = 4.0
    I_targ_val = 0.3855
    S_E_ss = (P['tau_E'] * P['gamma'] * r_targ) / (1 + P['tau_E'] * P['gamma'] * r_targ)
    S_I_ss = (P['tau_I'] * P['gamma'] * r_targ) / (1 + P['tau_I'] * P['gamma'] * r_targ)

    J_i = (P['W_E'] * P['I_0'] + P['w_plus'] * P['J_NMDA'] * S_E_ss + G * P[
        'J_NMDA'] * w_LRE * S_E_ss - I_targ_val) / S_I_ss
    J_i = max(0, J_i)

    # Init
    steps = int(P['T_sim'] / P['dt'])
    S_E = np.full(2, S_E_ss)
    S_I = np.full(2, S_I_ss)

    # X_BW: [s, f, v, q] for 2 nodes
    X_BW = np.array([[0.0, 1.0, 1.0, 1.0],
                     [0.0, 1.0, 1.0, 1.0]])

    # Buffer
    TR_steps = int(P['TR'] / P['dt'])
    bold_length = steps // TR_steps
    BOLD = np.zeros((2, bold_length))
    b_idx = 0

    sqrt_dt = np.sqrt(P['dt'])
    dt_sec = P['dt'] / 1000.0
    C_mat = np.array([[0.0, 1.0], [1.0, 0.0]])

    for k in range(1, steps + 1):
        noise = np.random.randn(2) * sigma * sqrt_dt

        # DMF
        # C_mat @ S_E effectively swaps the elements for the 2-node interaction
        I_E = P['W_E'] * P['I_0'] + P['w_plus'] * P['J_NMDA'] * S_E + G * P['J_NMDA'] * (
                    w_LRE * (C_mat @ S_E)) - J_i * S_I
        I_I = P['W_I'] * P['I_0'] + P['J_NMDA'] * S_E + G * P['J_NMDA'] * (w_FFI * (C_mat @ S_E)) - S_I

        # Safe Excitatory Rate
        num_E = P['a_E'] * I_E - P['b_E']
        r_E = np.zeros_like(I_E)
        mask_E = np.abs(num_E) > 1e-9
        r_E[mask_E] = num_E[mask_E] / (1 - np.exp(-P['d_E'] * num_E[mask_E]))
        r_E[r_E < 0] = 0

        # Safe Inhibitory Rate
        num_I = P['a_I'] * I_I - P['b_I']
        r_I = np.zeros_like(I_I)
        mask_I = np.abs(num_I) > 1e-9
        r_I[mask_I] = num_I[mask_I] / (1 - np.exp(-P['d_I'] * num_I[mask_I]))
        r_I[r_I < 0] = 0

        S_E = S_E + P['dt'] * (-S_E / P['tau_E'] + (1 - S_E) * P['gamma'] * r_E + noise)
        S_I = S_I + P['dt'] * (-S_I / P['tau_I'] + P['gamma'] * r_I + noise)

        # Balloon-Windkessel
        s = X_BW[:, 0];
        f = X_BW[:, 1];
        v = X_BW[:, 2];
        q = X_BW[:, 3]

        ds = S_E - s / BW['tau_s'] - (f - 1) / BW['tau_f']
        df = s
        dv = (f - v ** (1 / BW['alpha'])) / BW['tau_0']
        dq = (f * (1 - (1 - BW['E0']) ** (1 / f)) / BW['E0'] - q * v ** (1 / BW['alpha'] - 1)) / BW['tau_0']

        X_BW[:, 0] += ds * dt_sec
        X_BW[:, 1] += df * dt_sec
        X_BW[:, 2] += dv * dt_sec
        X_BW[:, 3] += dq * dt_sec

        if k % TR_steps == 0:
            if b_idx < bold_length:
                # Update BOLD
                y = BW['V0'] * (BW['k1'] * (1 - q) + BW['k2'] * (1 - q / v) + BW['k3'] * (1 - v))
                BOLD[:, b_idx] = y
                b_idx += 1

    washout = int(60000 / P['TR'])
    if bold_length > washout:
        R = np.corrcoef(BOLD[0, washout:], BOLD[1, washout:])
        fc = R[0, 1]
    else:
        fc = 0.0

    return fc


def run_trials_parallel(P, BW, w_LRE, w_FFI, G, sigma, trials):
    """Helper function to run the inner trial loop for a single parameter set."""
    acc = 0.0
    for t in range(trials):
        acc += simulate_trial(P, BW, w_LRE, w_FFI, G, sigma)
    return acc / trials


# =========================================================================
# SWEEP MANAGER
# =========================================================================
def run_sweep(P, BW, Ratios, VarParam, FixedParam, Mode):
    FC_Matrix = np.zeros((len(VarParam), len(Ratios)))

    for i in range(len(VarParam)):
        if Mode == 'Noise':
            sigma = VarParam[i]
            G = FixedParam
        else:
            G = VarParam[i]
            sigma = FixedParam

        print(f"   Simulating {Mode} level {i + 1}/{len(VarParam)}...")

        # Using ProcessPoolExecutor to mimic MATLAB's parfor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for ratio in Ratios:
                w_FFI = 1.0 / (1.0 + ratio)
                w_LRE = ratio * w_FFI
                futures.append(executor.submit(run_trials_parallel, P, BW, w_LRE, w_FFI, G, sigma, P['trials']))

            for r, future in enumerate(futures):
                FC_Matrix[i, r] = future.result()

    return FC_Matrix


# =========================================================================
# VISUALIZATION "TRICK" (SPLINE INTERPOLATION)
# =========================================================================
def format_panel(ax, TitleStr, LegendVals):
    ax.set_xscale('log')
    ax.set_xlim([0.01, 100])
    ax.set_ylim([-1.05, 1.05])
    ax.axhline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel('E/I Ratio', fontsize=12)
    ax.set_ylabel('FC', fontsize=12)
    ax.set_title(TitleStr, fontsize=12, fontweight='bold')
    ax.legend([str(val) for val in LegendVals], loc='lower right', frameon=False)
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.set_box_aspect(1)  # Equivalant to axis square

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)


def plot_smooth_curves(X_Data, Y_Noise, Y_Coup, Noises, Couplings):
    # Position [100 100 500 800] is roughly 5x8 inches
    fig = plt.figure(figsize=(5, 8), facecolor='w')

    # High-resolution X-axis for interpolation
    # Working in log space for the interpolation to match the visual scale
    log_X_Data = np.log10(X_Data)
    log_X_Smooth = np.linspace(np.min(log_X_Data), np.max(log_X_Data), 200)
    X_Smooth = 10 ** log_X_Smooth

    # Colors matched to paper (Blue -> Green -> Red -> Grey)
    cmap = np.array([
        [0.0, 0.0, 0.6],
        [0.2, 0.6, 1.0],
        [0.9, 0.7, 0.0],
        [0.0, 0.6, 0.0],
        [0.8, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    idx_noise = [0, 1, 2, 3, 4]  # 0-indexed
    idx_coup = [0, 1, 2, 4, 5]  # 0-indexed

    # --- Panel A: Noise ---
    ax1 = plt.subplot(2, 1, 1)
    for i in range(len(Noises)):
        # The "Trick": Spline Interpolation (Akima matches makima well)
        interpolator = Akima1DInterpolator(log_X_Data, Y_Noise[i, :])
        Y_Smooth = interpolator(log_X_Smooth)

        ax1.plot(X_Smooth, Y_Smooth, linewidth=2.5, color=cmap[idx_noise[i]])
    format_panel(ax1, r'Effect of Noise $\sigma$', Noises)

    # --- Panel B: Coupling ---
    ax2 = plt.subplot(2, 1, 2)
    for i in range(len(Couplings)):
        interpolator = Akima1DInterpolator(log_X_Data, Y_Coup[i, :])
        Y_Smooth = interpolator(log_X_Smooth)

        ax2.plot(X_Smooth, Y_Smooth, linewidth=2.5, color=cmap[idx_coup[i]])
    format_panel(ax2, 'Effect of Coupling G', Couplings)

    plt.tight_layout()
    plt.show()


# =========================================================================
# MAIN SCRIPT EXECUTION
# =========================================================================
if __name__ == '__main__':
    # --- 1. Global Parameters ---
    P = {
        'a_E': 310, 'b_E': 125, 'd_E': 0.16,
        'a_I': 615, 'b_I': 177, 'd_I': 0.087,
        'tau_E': 100, 'tau_I': 10,  # (ms)
        'gamma': 0.641 / 1000,
        'w_plus': 1.4,
        'J_NMDA': 0.15,
        'W_E': 1.0, 'W_I': 0.7,
        'I_0': 0.382,
        'dt': 1.0,
        'T_sim': 4 * 60 * 1000,  # 4 mins
        'TR': 720,
        'trials': 100  # Averaging factor
    }

    # --- Balloon-Windkessel (Friston 2003 Parameters) ---
    BW = {
        'tau_s': 0.65, 'tau_f': 0.41, 'tau_0': 0.98, 'alpha': 0.32,
        'E0': 0.34, 'V0': 0.02
    }
    BW['k1'] = 7 * BW['E0']
    BW['k2'] = 2.0
    BW['k3'] = 2 * BW['E0'] - 0.2

    # --- 2. Conditions ---
    # Use fewer points for simulation (speed), but interpolate many for plot
    EI_Ratios_Sim = np.logspace(-2, 2, 12)

    # Panel A: Noise Sweep
    Noises = [0.001, 0.005, 0.01, 0.025, 0.05]
    G_Fixed = 1.0

    # Panel B: Coupling Sweep
    Couplings = [0.01, 0.1, 0.2, 0.5, 1.0]
    Noise_Fixed = 0.01

    # --- 3. Run Simulation ---
    print('Running Simulation (Smoothing active)...')
    # Guard against accidental recursive spawning on Windows
    FC_Noise = run_sweep(P, BW, EI_Ratios_Sim, Noises, G_Fixed, 'Noise')
    FC_Coup = run_sweep(P, BW, EI_Ratios_Sim, Couplings, Noise_Fixed, 'Coupling')

    # --- 4. Plotting with Spline Interpolation ---
    plot_smooth_curves(EI_Ratios_Sim, FC_Noise, FC_Coup, Noises, Couplings)