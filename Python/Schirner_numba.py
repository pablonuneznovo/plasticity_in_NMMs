import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from numba import njit, prange


# =========================================================================
# NUMBA COMPILED MATH BLOCK
# =========================================================================
@njit(fastmath=True)
def transfer_function(I, a, b, d):
    """JIT-compiled transfer function for safe firing rates."""
    num = a * I - b
    if abs(num) < 1e-9:
        r = 1.0 / d
    else:
        r = num / (1.0 - np.exp(-d * num))
    return r if r > 0.0 else 0.0


@njit(fastmath=True)
def pearson_corr(x, y):
    """JIT-compiled 1D Pearson correlation."""
    xm = x - np.mean(x)
    ym = y - np.mean(y)
    num = np.sum(xm * ym)
    den = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    return num / den if den != 0.0 else 0.0


@njit(parallel=True, fastmath=True)
def run_sweep_jit(Ratios, VarParam, FixedParam, Mode_is_Noise,
                  P_a_E, P_b_E, P_d_E, P_a_I, P_b_I, P_d_I,
                  P_tau_E, P_tau_I, P_gamma, P_w_plus, P_J_NMDA,
                  P_W_E, P_W_I, P_I_0, P_dt, P_T_sim, P_TR, P_trials,
                  BW_tau_s, BW_tau_f, BW_tau_0, BW_alpha, BW_E0, BW_V0,
                  BW_k1, BW_k2, BW_k3):

    num_var = len(VarParam)
    num_rat = len(Ratios)
    FC_Matrix = np.zeros((num_var, num_rat))

    # Precompute constants
    r_targ = 4.0
    I_targ_val = 0.3855
    S_E_ss = (P_tau_E * P_gamma * r_targ) / (1.0 + P_tau_E * P_gamma * r_targ)
    S_I_ss = (P_tau_I * P_gamma * r_targ) / (1.0 + P_tau_I * P_gamma * r_targ)

    steps = int(P_T_sim / P_dt)
    TR_steps = int(P_TR / P_dt)
    bold_length = steps // TR_steps
    washout = int(60000 / P_TR)
    sqrt_dt = np.sqrt(P_dt)
    dt_sec = P_dt / 1000.0

    # Loop over the variable parameters (Noise or Coupling)
    for i in range(num_var):
        sigma = VarParam[i] if Mode_is_Noise else FixedParam
        G = FixedParam if Mode_is_Noise else VarParam[i]

        # Parallel loop over the E/I ratios
        for r in prange(num_rat):
            ratio = Ratios[r]
            w_FFI = 1.0 / (1.0 + ratio)
            w_LRE = ratio * w_FFI

            J_i = (P_W_E * P_I_0 + P_w_plus * P_J_NMDA * S_E_ss + G * P_J_NMDA * w_LRE * S_E_ss - I_targ_val) / S_I_ss
            if J_i < 0: J_i = 0.0

            acc = 0.0
            # Averaging loop
            for t in range(P_trials):
                S_E_0, S_E_1 = S_E_ss, S_E_ss
                S_I_0, S_I_1 = S_I_ss, S_I_ss
                s_0, s_1 = 0.0, 0.0
                f_0, f_1 = 1.0, 1.0
                v_0, v_1 = 1.0, 1.0
                q_0, q_1 = 1.0, 1.0

                BOLD_0 = np.zeros(bold_length)
                BOLD_1 = np.zeros(bold_length)
                b_idx = 0

                # 4-minute Integration loop
                for k in range(1, steps + 1):
                    noise_0 = np.random.randn() * sigma * sqrt_dt
                    noise_1 = np.random.randn() * sigma * sqrt_dt

                    I_E_0 = P_W_E * P_I_0 + P_w_plus * P_J_NMDA * S_E_0 + G * P_J_NMDA * (w_LRE * S_E_1) - J_i * S_I_0
                    I_E_1 = P_W_E * P_I_0 + P_w_plus * P_J_NMDA * S_E_1 + G * P_J_NMDA * (w_LRE * S_E_0) - J_i * S_I_1

                    I_I_0 = P_W_I * P_I_0 + P_J_NMDA * S_E_0 + G * P_J_NMDA * (w_FFI * S_E_1) - S_I_0
                    I_I_1 = P_W_I * P_I_0 + P_J_NMDA * S_E_1 + G * P_J_NMDA * (w_FFI * S_E_0) - S_I_1

                    r_E_0 = transfer_function(I_E_0, P_a_E, P_b_E, P_d_E)
                    r_E_1 = transfer_function(I_E_1, P_a_E, P_b_E, P_d_E)
                    r_I_0 = transfer_function(I_I_0, P_a_I, P_b_I, P_d_I)
                    r_I_1 = transfer_function(I_I_1, P_a_I, P_b_I, P_d_I)

                    S_E_0 += P_dt * (-S_E_0 / P_tau_E + (1.0 - S_E_0) * P_gamma * r_E_0 + noise_0)
                    S_E_1 += P_dt * (-S_E_1 / P_tau_E + (1.0 - S_E_1) * P_gamma * r_E_1 + noise_1)

                    S_I_0 += P_dt * (-S_I_0 / P_tau_I + P_gamma * r_I_0 + noise_0)
                    S_I_1 += P_dt * (-S_I_1 / P_tau_I + P_gamma * r_I_1 + noise_1)

                    # BW Update Node 0
                    ds0 = S_E_0 - s_0 / BW_tau_s - (f_0 - 1.0) / BW_tau_f
                    df0 = s_0
                    dv0 = (f_0 - v_0 ** (1.0 / BW_alpha)) / BW_tau_0
                    dq0 = (f_0 * (1.0 - (1.0 - BW_E0) ** (1.0 / f_0)) / BW_E0 - q_0 * v_0 ** (
                                1.0 / BW_alpha - 1.0)) / BW_tau_0
                    s_0 += ds0 * dt_sec;
                    f_0 += df0 * dt_sec;
                    v_0 += dv0 * dt_sec;
                    q_0 += dq0 * dt_sec

                    # BW Update Node 1
                    ds1 = S_E_1 - s_1 / BW_tau_s - (f_1 - 1.0) / BW_tau_f
                    df1 = s_1
                    dv1 = (f_1 - v_1 ** (1.0 / BW_alpha)) / BW_tau_0
                    dq1 = (f_1 * (1.0 - (1.0 - BW_E0) ** (1.0 / f_1)) / BW_E0 - q_1 * v_1 ** (
                                1.0 / BW_alpha - 1.0)) / BW_tau_0
                    s_1 += ds1 * dt_sec;
                    f_1 += df1 * dt_sec;
                    v_1 += dv1 * dt_sec;
                    q_1 += dq1 * dt_sec

                    if k % TR_steps == 0:
                        if b_idx < bold_length:
                            BOLD_0[b_idx] = BW_V0 * (
                                        BW_k1 * (1.0 - q_0) + BW_k2 * (1.0 - q_0 / v_0) + BW_k3 * (1.0 - v_0))
                            BOLD_1[b_idx] = BW_V0 * (
                                        BW_k1 * (1.0 - q_1) + BW_k2 * (1.0 - q_1 / v_1) + BW_k3 * (1.0 - v_1))
                            b_idx += 1

                if bold_length > washout:
                    fc = pearson_corr(BOLD_0[washout:], BOLD_1[washout:])
                else:
                    fc = 0.0

                acc += fc
            FC_Matrix[i, r] = acc / P_trials

    return FC_Matrix


# =========================================================================
# PYTHON WRAPPER
# =========================================================================
def run_sweep(P, BW, Ratios, VarParam, FixedParam, Mode):
    Mode_is_Noise = (Mode == 'Noise')

    return run_sweep_jit(
        Ratios, VarParam, FixedParam, Mode_is_Noise,
        P['a_E'], P['b_E'], P['d_E'], P['a_I'], P['b_I'], P['d_I'],
        P['tau_E'], P['tau_I'], P['gamma'], P['w_plus'], P['J_NMDA'],
        P['W_E'], P['W_I'], P['I_0'], P['dt'], P['T_sim'], P['TR'], P['trials'],
        BW['tau_s'], BW['tau_f'], BW['tau_0'], BW['alpha'], BW['E0'], BW['V0'],
        BW['k1'], BW['k2'], BW['k3']
    )


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
    ax.set_box_aspect(1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)


def plot_smooth_curves(X_Data, Y_Noise, Y_Coup, Noises, Couplings):
    fig = plt.figure(figsize=(5, 8), facecolor='w')

    log_X_Data = np.log10(X_Data)
    log_X_Smooth = np.linspace(np.min(log_X_Data), np.max(log_X_Data), 200)
    X_Smooth = 10 ** log_X_Smooth

    cmap = np.array([
        [0.0, 0.0, 0.6],
        [0.2, 0.6, 1.0],
        [0.9, 0.7, 0.0],
        [0.0, 0.6, 0.0],
        [0.8, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    idx_noise = [0, 1, 2, 3, 4]
    idx_coup = [0, 1, 2, 4, 5]

    # --- Panel A: Noise ---
    ax1 = plt.subplot(2, 1, 1)
    for i in range(len(Noises)):
        interpolator = Akima1DInterpolator(log_X_Data, Y_Noise[i, :])
        ax1.plot(X_Smooth, interpolator(log_X_Smooth), linewidth=2.5, color=cmap[idx_noise[i]])
    format_panel(ax1, r'Effect of Noise $\sigma$', Noises)

    # --- Panel B: Coupling ---
    ax2 = plt.subplot(2, 1, 2)
    for i in range(len(Couplings)):
        interpolator = Akima1DInterpolator(log_X_Data, Y_Coup[i, :])
        ax2.plot(X_Smooth, interpolator(log_X_Smooth), linewidth=2.5, color=cmap[idx_coup[i]])
    format_panel(ax2, 'Effect of Coupling G', Couplings)

    plt.tight_layout()
    plt.show()


# =========================================================================
# MAIN SCRIPT EXECUTION
# =========================================================================
if __name__ == '__main__':
    # --- 1. Global Parameters ---
    P = {
        'a_E': 310.0, 'b_E': 125.0, 'd_E': 0.16,
        'a_I': 615.0, 'b_I': 177.0, 'd_I': 0.087,
        'tau_E': 100.0, 'tau_I': 10.0,  # (ms)
        'gamma': 0.641 / 1000.0,
        'w_plus': 1.4,
        'J_NMDA': 0.15,
        'W_E': 1.0, 'W_I': 0.7,
        'I_0': 0.382,
        'dt': 1.0,
        'T_sim': 4.0 * 60.0 * 1000.0,  # 4 mins
        'TR': 720.0,
        'trials': 100  # Averaging factor
    }

    # --- Balloon-Windkessel (Friston 2003 Parameters) ---
    BW = {
        'tau_s': 0.65, 'tau_f': 0.41, 'tau_0': 0.98, 'alpha': 0.32,
        'E0': 0.34, 'V0': 0.02
    }
    BW['k1'] = 7.0 * BW['E0']
    BW['k2'] = 2.0
    BW['k3'] = 2.0 * BW['E0'] - 0.2

    # --- 2. Conditions ---
    EI_Ratios_Sim = np.logspace(-2, 2, 12)

    Noises = np.array([0.001, 0.005, 0.01, 0.025, 0.05])
    G_Fixed = 1.0

    Couplings = np.array([0.01, 0.1, 0.2, 0.5, 1.0])
    Noise_Fixed = 0.01

    # --- 3. Run Simulation ---
    print('Compiling Numba functions (this takes a few seconds)...')
    # The first time these run, Numba compiles them. Subsequent runs are instantaneous.
    print('Running Noise Sweep...')
    FC_Noise = run_sweep(P, BW, EI_Ratios_Sim, Noises, G_Fixed, 'Noise')
    print('Running Coupling Sweep...')
    FC_Coup = run_sweep(P, BW, EI_Ratios_Sim, Couplings, Noise_Fixed, 'Coupling')

    # --- 4. Plotting ---
    plot_smooth_curves(EI_Ratios_Sim, FC_Noise, FC_Coup, Noises, Couplings)