import numpy as np
import os
os.chdir('/home/yunwei/SNAIL/transmon/monte_carlo/new/full_simulation')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from hamiltonian_generator import Hamiltonian
from system import *
import qutip as qt
import pickle
import warnings


def fit_T1_function(time_points, population_data):
    """Fits y = 1 - exp(-t/T1), returns T1."""
    def fit1(t, T1):
        return 1 - np.exp(-t / T1)
    try:
        popt, _ = curve_fit(fit1, time_points, population_data, p0=[np.median(time_points)])
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"T1 fit failed: {e}")
        return np.nan


def fit_thermal_gamma_up(time_points, population_data, gamma_down=1/(2e4), N=0.5):
    """Fits N_2(t) = N * gamma_up/(gamma_up+gamma_down) * (1 - exp(-(gamma_up+gamma_down)*t)).
    Returns gamma_up."""
    def thermal_model(t, gamma_up):
        return N * gamma_up / (gamma_up + gamma_down) * (1 - np.exp(-(gamma_up + gamma_down) * t))
    try:
        popt, _ = curve_fit(thermal_model, time_points, population_data,
                            p0=[1e-6], bounds=(0, np.inf))
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Thermal gamma_up fit failed: {e}")
        return np.nan


def fit_decay_function(time_points, population_data):
    """Fits y = exp(-t/T), returns T."""
    def decay(t, T):
        return np.exp(-t / T)
    try:
        popt, _ = curve_fit(decay, time_points, population_data, p0=[np.median(time_points)])
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Decay fit failed: {e}")
        return np.nan


def fit_T2_function(time_points, population_data, omega):
    """Fits y = exp(-t/T2) * cos(omega*t), returns T2."""
    def T2fit(t, T2):
        return np.exp(-t / T2) * np.cos(omega * t)
    try:
        popt, _ = curve_fit(T2fit, time_points, population_data, p0=[1e6])
        return popt[0]
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"T2 fit failed: {e}")
        return np.nan


def reconstruct_sim(A):
    """Reconstruct a lightweight Simulation-like object for Floquet projectors."""
    phi_ex = 0.2
    Ej = 30.19
    Ec = 0.1
    sc = Hamiltonian(phi_ex, Ej, Ec, [5, 3])
    optimal_omega, _ = sc.optimal_omegad(A)
    optimal_omega = optimal_omega * 2 * np.pi
    sc = Hamiltonian(phi_ex, Ej, Ec, [3, 2])
    _, kick_and_sigmax, get_projector = sc.setup_floquet_system(A, optimal_omega)
    return kick_and_sigmax, get_projector


def compute_projector_expectations(avg_rho, time_points, kick_and_sigmax, get_projector):
    """Compute expectation values for all projectors from average density matrix."""
    floquet_states_0, _, _ = kick_and_sigmax(0)
    num_states = floquet_states_0.shape[1]
    expectations = {k: np.zeros(len(time_points)) for k in range(num_states)}

    for j, t in enumerate(time_points):
        floquet_states, _, _ = kick_and_sigmax(t)
        projectors = get_projector(floquet_states)
        rho_j = qt.Qobj(avg_rho[j])
        for k in range(num_states):
            expectations[k][j] = np.real(qt.expect(projectors[k], rho_j))
    return expectations


def compute_sigmax_expectation(avg_rho, time_points, kick_and_sigmax):
    """Compute <sigmax> from average density matrix."""
    sigmax_vals = np.zeros(len(time_points))
    for j, t in enumerate(time_points):
        _, sigmax, _ = kick_and_sigmax(t)
        rho_j = qt.Qobj(avg_rho[j])
        sigmax_vals[j] = np.real(qt.expect(qt.Qobj(sigmax), rho_j))
    return sigmax_vals


def state_label(k, trunc=(3, 2)):
    """Map flat index k to (transmon, cavity) label string."""
    cavity_dim = trunc[1]
    t_idx = k // cavity_dim
    c_idx = k % cavity_dim
    return f"{t_idx}{c_idx}"


if __name__ == '__main__':
    DATA_FILE = 'total_rate_debug_data.pkl'

    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    A = data['A']
    S0_values = data['S0_values']
    results = data['results']

    kick_and_sigmax, get_projector = reconstruct_sim(A)

    for S0 in S0_values:
        if S0 not in results:
            continue
        r = results[S0]
        print(f"\n{'='*60}")
        print(f"S0 = {S0:.2e}")
        print(f"{'='*60}")

        # --- T1 analysis: fit growth of P[0] ---
        t1_projs = compute_projector_expectations(
            r['avg_rho_T1'], r['t1_time_points'], kick_and_sigmax, get_projector
        )
        P0 = t1_projs[0]
        T1 = fit_T1_function(r['t1_time_points'], P0)
        print(f"  T1 = {T1:.2f} ns")

        # --- T2 analysis: sigmax expectation ---
        sigmax_vals = compute_sigmax_expectation(
            r['avg_rho_T2'], r['t2_time_points'], kick_and_sigmax
        )
        fit_omega = r['fit_omega']
        T2 = fit_T2_function(r['t2_time_points'], sigmax_vals, fit_omega)
        print(f"  T2 = {T2:.2f} ns")

        # --- Dephasing rate ---
        if T1 > 0 and T2 > 0 and not (np.isnan(T1) or np.isnan(T2)):
            dephasing_rate = 1/T2 - 1/(2*T1)
            print(f"  1/T_phi = 1/T2 - 1/(2*T1) = {dephasing_rate:.6e}")
        else:
            print(f"  Could not compute dephasing rate (T1 or T2 fit failed)")

        # --- Print all projector expectations at final time ---
        print(f"\n  T1 projector expectations (t=0 -> t={r['t1_time_points'][-1]:.0f}):")
        for k in sorted(t1_projs.keys()):
            print(f"    P[{state_label(k)}]: {t1_projs[k][0]:.6f} -> {t1_projs[k][-1]:.6f}")

        t2_projs = compute_projector_expectations(
            r['avg_rho_T2'], r['t2_time_points'], kick_and_sigmax, get_projector
        )
        print(f"\n  T2 projector expectations (t=0 -> t={r['t2_time_points'][-1]:.0f}):")
        for k in sorted(t2_projs.keys()):
            print(f"    P[{state_label(k)}]: {t2_projs[k][0]:.6f} -> {t2_projs[k][-1]:.6f}")

        # --- |10⟩ transmon decay analysis ---
        gamma_down = 1 / (2e4)  # default
        if 'avg_rho_10' in r:
            t10_projs = compute_projector_expectations(
                r['avg_rho_10'], r['t10_time_points'], kick_and_sigmax, get_projector
            )
            P10 = t10_projs[2]  # state |10⟩
            T_transmon = fit_decay_function(r['t10_time_points'], P10)
            gamma_down = 1 / T_transmon
            print(f"\n  Transmon decay (from |10⟩):")
            print(f"    T_transmon = {T_transmon:.2f} ns")
            print(f"    γ↓ = 1/T_transmon = {gamma_down:.6e} /ns")
            print(f"    |10⟩ projector expectations (t=0 -> t={r['t10_time_points'][-1]:.0f}):")
            for k in sorted(t10_projs.keys()):
                print(f"      P[{state_label(k)}]: {t10_projs[k][0]:.6f} -> {t10_projs[k][-1]:.6f}")

        # --- P[10] thermal fit from T2 data (using γ↓ from transmon decay) ---
        P10_t2 = t2_projs[2]
        gamma_up = fit_thermal_gamma_up(r['t2_time_points'], P10_t2, gamma_down=gamma_down, N=0.5)
        print(f"\n  P[10] thermal fit from T2 data (γ↓={gamma_down:.2e} from transmon decay, N=1/2):")
        print(f"    γ↑ = {gamma_up:.6e} /ns")
        if not np.isnan(gamma_up):
            P10_ss = 0.5 * gamma_up / (gamma_up + gamma_down)
            print(f"    P[10] predicted t→∞ = {P10_ss:.6e}")
            print(f"    P[10] at end of sim = {P10_t2[-1]:.6e}")
            print(f"    γ↑ + γ↓ = {gamma_up + gamma_down:.6e} /ns")

    # --- Plots for last S0 ---
    S0 = S0_values[-1]
    if S0 in results:
        r = results[S0]
        t1_projs = compute_projector_expectations(
            r['avg_rho_T1'], r['t1_time_points'], kick_and_sigmax, get_projector
        )
        num_states = len(t1_projs)

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        # (0,0) T1: P[00] growth data vs fit
        P0 = t1_projs[0]
        T1 = fit_T1_function(r['t1_time_points'], P0)
        t1_arr = r['t1_time_points']
        P0_fit = 1 - np.exp(-t1_arr / T1)
        axes[0, 0].plot(t1_arr, P0, label='P[00] (data)')
        axes[0, 0].plot(t1_arr, P0_fit, '--', color='red', label=f'fit: T1={T1:.0f} ns')
        axes[0, 0].set_xlabel('Time (ns)', fontsize=12)
        axes[0, 0].set_ylabel('Population', fontsize=12)
        axes[0, 0].set_title(f'T1: P[00] vs fit (S0={S0:.1e})', fontsize=13)
        axes[0, 0].legend(fontsize=11)

        # (0,1) T2: sigmax data vs fit
        sigmax_vals = compute_sigmax_expectation(
            r['avg_rho_T2'], r['t2_time_points'], kick_and_sigmax
        )
        fit_omega = r['fit_omega']
        T2 = fit_T2_function(r['t2_time_points'], sigmax_vals, fit_omega)
        t2_arr = r['t2_time_points']
        sigmax_fit = np.exp(-t2_arr / T2) * np.cos(fit_omega * t2_arr)
        axes[0, 1].plot(t2_arr, sigmax_vals, label=r'$\langle\sigma_x\rangle$ (data)')
        axes[0, 1].plot(t2_arr, sigmax_fit, '--', color='red', label=f'fit: T2={T2:.0f} ns')
        axes[0, 1].set_xlabel('Time (ns)', fontsize=12)
        axes[0, 1].set_ylabel(r'$\langle\sigma_x\rangle$', fontsize=12)
        axes[0, 1].set_title(f'T2: sigmax vs fit (S0={S0:.1e})', fontsize=13)
        axes[0, 1].legend(fontsize=11)

        # (0,2) |10⟩ transmon decay: P[10] vs fit
        if 'avg_rho_10' in r:
            t10_projs = compute_projector_expectations(
                r['avg_rho_10'], r['t10_time_points'], kick_and_sigmax, get_projector
            )
            P10 = t10_projs[2]
            T_transmon = fit_decay_function(r['t10_time_points'], P10)
            t10_arr = r['t10_time_points']
            P10_fit = np.exp(-t10_arr / T_transmon)
            axes[0, 2].plot(t10_arr, P10, label='P[10] (data)')
            axes[0, 2].plot(t10_arr, P10_fit, '--', color='red', label=f'fit: T={T_transmon:.0f} ns')
            axes[0, 2].set_xlabel('Time (ns)', fontsize=12)
            axes[0, 2].set_ylabel('Population', fontsize=12)
            axes[0, 2].set_title(f'|10⟩ decay: P[10] vs fit (S0={S0:.1e})', fontsize=13)
            axes[0, 2].legend(fontsize=11)

        # (1,0) T2: all projectors
        t2_projs = compute_projector_expectations(
            r['avg_rho_T2'], r['t2_time_points'], kick_and_sigmax, get_projector
        )
        for k in range(num_states):
            axes[1, 0].plot(t2_arr, t2_projs[k], label=f'P[{state_label(k)}]')
        axes[1, 0].set_xlabel('Time (ns)', fontsize=12)
        axes[1, 0].set_ylabel('Population', fontsize=12)
        axes[1, 0].set_title(f'T2: all projectors (S0={S0:.1e})', fontsize=13)
        axes[1, 0].legend()

        # (1,1) T2: P[10] thermal growth with fit
        if 'avg_rho_10' in r:
            gamma_down = 1 / T_transmon
        else:
            gamma_down = 1 / (2e4)
        P10_t2 = t2_projs[2]
        gamma_up_t2 = fit_thermal_gamma_up(t2_arr, P10_t2, gamma_down=gamma_down, N=0.5)
        P10_t2_fit = 0.5 * gamma_up_t2 / (gamma_up_t2 + gamma_down) * (1 - np.exp(-(gamma_up_t2 + gamma_down) * t2_arr))
        axes[1, 1].plot(t2_arr, P10_t2, label='P[10] (data)')
        axes[1, 1].plot(t2_arr, P10_t2_fit, '--', color='red', label=f'fit: $\\gamma_\\uparrow$={gamma_up_t2:.2e}')
        axes[1, 1].set_xlabel('Time (ns)', fontsize=12)
        axes[1, 1].set_ylabel('Population', fontsize=12)
        axes[1, 1].set_title(f'T2: P[10] thermal fit (S0={S0:.1e})', fontsize=13)
        axes[1, 1].legend(fontsize=11)

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('debug_projector_dynamics.pdf')
        print(f"\nPlot saved to debug_projector_dynamics.pdf")
        plt.show()
