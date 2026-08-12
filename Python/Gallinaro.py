"""
Gallinaro et al. (2022) structural-plasticity mean-field figure.

This script translates the Mathematica notebook `time-series-fig4.nb` from:
https://github.com/juliavg/engram_structural_plasticity

It solves the deterministic mean-field ODE system used for Gallinaro et al.,
Fig. 4: population rates, calcium traces, and the 2-by-2 excitatory
connectivity matrix. Stochastic NEST connectivity samples distributed with
the authors' public code are loaded from `gallinaro_nest_connectivity_data.mat`
and overlaid as dots in panel C.

Dependencies: numpy, scipy, matplotlib.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.io import loadmat
from scipy.special import erfc, expit


# --- Colour Palette ---
c_rate = (0.2, 0.2, 0.2)
c_input = (0.6, 0.6, 0.6)
c_trace = (0.3, 0.5, 0.3)
c_facil = (0.8, 0.3, 0.3)
c_depress = (0.2, 0.4, 0.8)
c_memory = (0.6, 0.4, 0.8)
c_ax = (0.15, 0.15, 0.15)


def build_parameters():
    """Parameters from Gallinaro et al. Fig. 4 notebook."""
    p = {
        "J": 0.1,
        "tau_m": 0.02,
        "tau_r": 0.002,
        "theta": 20.0,
        "v_reset": 10.0,
        "g": 8.0,
        "epsilon": 0.1,
        "n_e1": 1000.0,
        "n_e2": 9000.0,
        "n_x": 10000.0,
        "nu_ext": 15.0,
        "n_i": 2500.0,
        "tau_rate": 0.0002,
        "tau_ca": 10.0,
        "beta_a": 2.0,
        "beta_d": 2.0,
        "k_a": 1.0,
        "k_d": 1.0,
        "t0": 0.0,
        "tf": 6150.0,
        "tf_plot": 1150.0,
        "t_on": 500.0,
        "t_off": 650.0,
        "target_rate": 8.0,
    }
    p["n_pop"] = np.array([p["n_e1"], p["n_e2"]], dtype=float)
    return p


def build_transfer_primitive():
    """
    Tabulate the primitive of exp(u^2) * erfc(-u), the Brunel/Siegert
    integral used in Gallinaro Eq. 13. This is a numerical acceleration of
    the same transfer function, not a replacement dynamics.
    """
    grid = np.linspace(-8.0, 8.0, 20001)
    integrand = np.exp(np.clip(grid * grid, None, 700.0)) * erfc(-grid)
    primitive = np.r_[0.0, cumulative_trapezoid(integrand, grid)]
    return grid, primitive


def primitive_value(x, grid, primitive):
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, grid[0], grid[-1])
    values = np.interp(x_clipped, grid, primitive)

    # Asymptotic correction for the negative tail of exp(u^2) * erfc(-u).
    # This matters only during the very early high-input growth transient.
    mask = x < grid[0]
    if np.any(mask):
        values = np.array(values, copy=True)
        values[mask] = -np.log((-x[mask]) / (-grid[0])) / np.sqrt(np.pi)
    return values


def brunel_rate(mu, sigma, p, grid, primitive):
    """Stationary LIF firing rate f(mu, sigma) used in Gallinaro Eq. 13."""
    sigma = np.maximum(sigma, 1e-9)
    a = (p["v_reset"] - mu) / sigma
    b = (p["theta"] - mu) / sigma

    integral = primitive_value(b, grid, primitive) - primitive_value(a, grid, primitive)
    rate = 1.0 / (p["tau_r"] + p["tau_m"] * np.sqrt(np.pi) * np.maximum(integral, 0.0))

    rate = np.where(a >= grid[-1], 0.0, rate)
    rate = np.where(b <= grid[0], 1.0 / p["tau_r"], rate)
    rate = np.where(b >= grid[-1], 0.0, rate)
    return rate


def stimulus(t, p):
    """Gallinaro Fig. 4 stimulation protocol: E1 receives +10% input."""
    gate = expit(1000.0 * (t - p["t_on"])) * expit(1000.0 * (p["t_off"] - t))
    if np.ndim(gate) == 0:
        return np.array([0.10 * gate, 0.0])
    return np.vstack((0.10 * gate, np.zeros_like(gate)))


def mean_field_rhs(t, y, p, grid, primitive):
    """
    State vector:
      phi_E1, phi_E2,
      W_E1E1, W_E1E2, W_E2E1, W_E2E2,
      r_E1, r_E2, r_I.
    """
    phi = np.maximum(y[0:2], 0.0)
    w = np.maximum(y[2:6].reshape(2, 2), 1e-12)
    rates = np.maximum(y[6:9], 0.0)
    r_e = rates[:2]
    r_i = rates[2]
    n_pop = p["n_pop"]

    # Eq. 15 ingredients: average in-degree and out-degree.
    k_in = np.maximum(w @ n_pop, 1e-12)
    k_out = np.maximum(w.T @ n_pop, 1e-12)

    # Eq. 15 is written here with positive deletion magnitudes and explicit
    # subtraction, matching the released Mathematica implementation.
    dend_create = np.maximum((p["target_rate"] - phi) * p["k_d"] / p["beta_d"], 0.0)
    axon_create = np.maximum((p["target_rate"] - phi) * p["k_a"] / p["beta_a"], 0.0)
    dend_delete = np.maximum((phi - p["target_rate"]) * p["k_d"] / p["beta_d"], 0.0)
    axon_delete = np.maximum((phi - p["target_rate"]) * p["k_a"] / p["beta_a"], 0.0)

    dendritic_loss = w * (dend_delete / k_in)[:, None]
    axonal_loss = w * (axon_delete / k_out)[None, :]

    free_dendrites = dend_create + (axonal_loss * n_pop[None, :]).sum(axis=1)
    free_axons = axon_create + (dendritic_loss * n_pop[:, None]).sum(axis=0)

    total_free_dendrites = (n_pop * free_dendrites).sum()
    total_free_axons = (n_pop * free_axons).sum()
    if total_free_dendrites > 0.0 and total_free_axons > 0.0:
        pairing_rate = min(total_free_dendrites, total_free_axons)
        lambda_pair = pairing_rate / (total_free_dendrites * total_free_axons)
    else:
        lambda_pair = 0.0

    dw_dt = lambda_pair * np.outer(free_dendrites, free_axons) - dendritic_loss - axonal_loss

    # Eq. 14: calcium/activity trace.
    dphi_dt = (r_e - phi) / p["tau_ca"]

    # Eq. 13: Wilson-Cowan relaxation to the stationary LIF transfer.
    e_input = (w @ (n_pop * r_e)) + (1.0 + stimulus(t, p)) * p["epsilon"] * p["n_x"] * p["nu_ext"]
    mu_e = p["tau_m"] * p["J"] * (e_input - p["g"] * p["epsilon"] * p["n_i"] * r_i)
    sigma_e = p["J"] * np.sqrt(np.maximum(p["tau_m"] * (e_input + p["g"] ** 2 * p["epsilon"] * p["n_i"] * r_i), 1e-12))
    target_e = brunel_rate(mu_e, sigma_e, p, grid, primitive)

    i_input = p["epsilon"] * np.sum(n_pop * r_e)
    mu_i = p["tau_m"] * p["J"] * (i_input - p["g"] * p["epsilon"] * p["n_i"] * r_i + p["epsilon"] * p["n_x"] * p["nu_ext"])
    sigma_i = p["J"] * np.sqrt(
        max(p["tau_m"] * (i_input + p["g"] ** 2 * p["epsilon"] * p["n_i"] * r_i + p["epsilon"] * p["n_x"] * p["nu_ext"]), 1e-12)
    )
    target_i = brunel_rate(mu_i, sigma_i, p, grid, primitive)
    dr_dt = (-rates + np.r_[target_e, target_i]) / p["tau_rate"]

    return np.r_[dphi_dt, dw_dt.ravel(), dr_dt]


def solve_mean_field():
    p = build_parameters()
    grid, primitive = build_transfer_primitive()

    initial = np.r_[
        np.ones(2) * p["target_rate"],
        np.ones(4) * 0.1,
        np.ones(3) * p["target_rate"],
    ]
    y0 = 0.001 * initial

    t_eval = np.linspace(p["t0"], p["tf"], 2500)
    sol = solve_ivp(
        lambda t, y: mean_field_rhs(t, y, p, grid, primitive),
        (p["t0"], p["tf"]),
        y0,
        method="LSODA",
        t_eval=t_eval,
        rtol=1e-5,
        atol=1e-7,
        max_step=2.0,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return p, sol.t, sol.y


def load_nest_connectivity():
    data_path = Path(__file__).resolve().parent / "gallinaro_nest_connectivity_data.mat"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path.name}. Keep it in the same folder as this script."
        )
    mat = loadmat(data_path, squeeze_me=True)
    time_s = np.asarray(mat["sim_steps"], dtype=float).ravel() / 1000.0
    data = {name: np.asarray(mat[name], dtype=float).ravel() for name in ["c11", "c12", "c21", "c22"]}
    return time_s, data


def make_figure():
    p, t, y = solve_mean_field()
    phi = y[0:2, :]
    w = y[2:6, :].T.reshape(-1, 2, 2)
    rates = y[6:9, :]
    input_e1 = stimulus(t, p)[0]
    nest_t, nest = load_nest_connectivity()

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "legend.frameon": False,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    fig.patch.set_facecolor("white")

    # --- Panel A: firing-rate dynamics and stimulation ---
    ax = axes[0]
    ax2 = ax.twinx()
    ax2.area = ax2.fill_between(t, 0.0, input_e1, color=c_input, alpha=0.2, linewidth=0)
    ax2.set_ylim(0.0, 0.15)
    ax2.set_ylabel(r"$s_{E_1}(t)$", color=c_input)
    ax2.tick_params(axis="y", colors=c_input)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(1.5)
    ax2.spines["right"].set_color(c_input)
    ax2.spines["top"].set_visible(False)

    ax.plot(t, rates[0], color=c_rate, linewidth=2.0, label=r"$r_{E_1}(t)$")
    ax.plot(t, rates[1], color=c_input, linewidth=1.6, label=r"$r_{E_2}(t)$")
    ax.axhline(p["target_rate"], color=c_rate, linestyle="--", linewidth=1.1, alpha=0.45)
    ax.set_ylabel(r"$r_Y(t)$ (Hz)", color=c_rate)
    ax.tick_params(axis="y", colors=c_rate)
    ax.set_ylim(0.0, max(rates[0].max(), rates[1].max()) * 1.18)
    ax.legend(loc="upper right", handlelength=1.8)

    # --- Panel B: calcium trace Eq. 14 ---
    ax = axes[1]
    ax.plot(t, phi[0], color=c_trace, linewidth=2.0, label=r"$\phi_{E_1}(t)$")
    ax.plot(t, phi[1], color=c_input, linewidth=1.6, label=r"$\phi_{E_2}(t)$")
    ax.axhline(p["target_rate"], color=c_rate, linestyle="--", linewidth=1.1, alpha=0.45)
    ax.set_ylabel(r"$\phi_Y(t)$")
    ax.set_ylim(0.0, max(phi[0].max(), phi[1].max()) * 1.14)
    ax.legend(loc="upper right", handlelength=1.8)

    # --- Panel C: connectivity matrix Eq. 15, with released NEST samples ---
    ax = axes[2]
    labels = [
        (0, 0, r"$\bar{C}_{E_1,E_1}(t)$", c_facil, "c11"),
        (0, 1, r"$\bar{C}_{E_1,E_2}(t)$", c_depress, "c12"),
        (1, 0, r"$\bar{C}_{E_2,E_1}(t)$", c_memory, "c21"),
        (1, 1, r"$\bar{C}_{E_2,E_2}(t)$", c_rate, "c22"),
    ]
    for row, col, label, color, data_key in labels:
        ax.plot(t, w[:, row, col], color=color, linewidth=2.0, label=label)
        ax.plot(
            nest_t,
            nest[data_key],
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            alpha=0.85,
        )
    ax.set_ylabel(r"$\bar{C}_{Y,Z}(t)$")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0.0, max(w[:, 0, 0].max(), nest["c11"].max()) * 1.08)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        handlelength=1.8,
        columnspacing=0.9,
    )

    for axis in axes:
        axis.axvline(p["t_on"], color=c_ax, linewidth=1.0, alpha=0.65)
        axis.axvline(p["t_off"], color=c_ax, linewidth=1.0, alpha=0.65)
        axis.set_xlim(p["t0"], p["tf_plot"])
        axis.set_xticks([0, 50, 200, 350, 500, 650, 800, 950, 1100])
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=12)

    fig.tight_layout(h_pad=1.15)


if __name__ == "__main__":
    make_figure()
