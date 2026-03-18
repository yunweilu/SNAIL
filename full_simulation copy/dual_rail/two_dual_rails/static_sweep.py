import pickle
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from system import Hamiltonian


def _style_setup():
    style_path = Path(__file__).resolve().parent.parent / "plot_instruction"
    style_mod = SourceFileLoader("plot_instruction_mod_dual_rail", str(style_path)).load_module()
    mpl.rcParams.update(style_mod.normal_plot)
    return style_mod


def _get_array(data, candidates, required=True):
    for key in candidates:
        if key in data:
            return np.array(data[key])
    if required:
        raise KeyError(f"Missing required key. Tried: {candidates}")
    return None


def _extract_series(data):
    # Primary expected keys
    d_c1 = _get_array(data, ["d_c1_arr", "d_c1", "dc1"])
    d_c2 = _get_array(data, ["d_c2_arr", "d_c2", "dc2"])

    # X-axis detuning (GHz preferred internally)
    detunings_ghz = _get_array(data, ["detunings_ghz"], required=False)
    if detunings_ghz is None:
        detunings_mhz = _get_array(data, ["detunings_mhz"], required=False)
        if detunings_mhz is not None:
            detunings_ghz = detunings_mhz * 1e-3

    # Fallback if only omega sweep and reference frequency are stored
    if detunings_ghz is None and "omega_ds" in data and "omega_s" in data:
        omega_ds = np.array(data["omega_ds"])
        omega_s = float(data["omega_s"])
        detunings_ghz = (omega_ds - omega_s) / (2 * np.pi)

    if detunings_ghz is None:
        raise KeyError("Could not infer detuning axis. Need detunings_ghz/detunings_mhz or omega_ds+omega_s.")

    return detunings_ghz, d_c1, d_c2


def _extract_norm(data):
    for key in [
        "domega_b_dphi",
        "d_omega_b_dphi",
        "domega_b_dphi_fd",
        "omega_b_phi_derivative",
        "omega_b_derivative",
    ]:
        if key in data:
            return float(data[key])
    raise KeyError(
        "Missing normalization derivative |d omega_b / dPhi| in data. "
        "Expected one of: domega_b_dphi, d_omega_b_dphi, domega_b_dphi_fd, "
        "omega_b_phi_derivative, omega_b_derivative."
    )


def _compute_norm_from_model(data=None):
    """Compute two cavities quasi-energy derivatives and sweep over drive frequency."""
    if data is None:
        data = {}
    Ej = float(data.get("Ej", 30.19))
    Ec = float(data.get("Ec", 0.1))
    omega_c1 = float(data.get("omega_c1", 5.226))
    omega_c2 = float(data.get("omega_c2", 7.335))
    phi_ex = float(data.get("phi_ex", 0.2))
    bare_dim = data.get("bare_dim", [10, 6, 6])
    trunc_dim = data.get("trunc_dim", [5, 2, 2])
    g_val = float(data.get("g_val", 0.05 * 2 * np.pi))

    def build(phi):
        sc = Hamiltonian(phi, Ej, Ec, bare_dim, trunc_dim, omega_c1, omega_c2)
        sc.g = g_val
        sc.H, sc.H_control, sc.H_flux_drive, sc.noise, sc.s = sc.get_H()
        sc.H_dressed, sc.H_control_dressed, sc.H_flux_drive_dressed = sc.dressed_basis()
        return sc

    # Base Hamiltonian
    sc0 = build(phi_ex)
    dim = sc0.original_dim
    idx_0 = sc0.state_index((0, 0, 0), dim)
    idx_b = sc0.state_index((1, 0, 0), dim)
    idx_c2 = sc0.state_index((0, 1, 0), dim)
    idx_c1 = sc0.state_index((0, 0, 1), dim)

    # Transmon (dressed) frequency
    omega_q = sc0.H_dressed[idx_b, idx_b].real - sc0.H_dressed[idx_0, idx_0].real

    # Detuning: drive freq - transmon freq (-50 MHz to 50 MHz)
    detunings_mhz = np.linspace(-50, 50, 201)
    drive_freqs = omega_q + (detunings_mhz * 1e-3) * 2 * np.pi
    
    # Drive amplitude 10 MHz
    A = 10e-3 * 2 * np.pi

    delta_phi = 1e-6
    sc_p = build(phi_ex + delta_phi)
    sc_m = build(phi_ex - delta_phi)

    derivs_c1 = []
    derivs_c2 = []
    
    for omegad in drive_freqs:
        E_c1_p, E_c2_p = sc_p.quasi_energy(A, omegad)
        E_c1_m, E_c2_m = sc_m.quasi_energy(A, omegad)
        
        grad_c1 = (E_c1_p - E_c1_m) / (2 * delta_phi) / (2 * np.pi)
        grad_c2 = (E_c2_p - E_c2_m) / (2 * delta_phi) / (2 * np.pi)
        
        derivs_c1.append(grad_c1)
        derivs_c2.append(grad_c2)

    return detunings_mhz, np.array(derivs_c1), np.array(derivs_c2)


def main():
    print("Computing quasi-energy derivatives...")
    detuning, dc1, dc2 = _compute_norm_from_model({})
    print("Done computing. Here are the results:")
    for d, d1, d2 in zip(detuning, dc1, dc2):
        print(f"Detuning: {d:>.1f} MHz | dc1/dPhi: {d1:>.6f} GHz/Phi0 | dc2/dPhi: {d2:>.6f} GHz/Phi0")

    try:
        _style_setup()
    except Exception as e:
        print(f"Plot style setup failed (continuing with default): {e}")

    plt.figure(figsize=(8, 6))
    plt.plot(detuning, np.abs(dc1), 'o-', label=r"Cavity 1 ($|dc_1/d\Phi|$)")
    plt.plot(detuning, np.abs(dc2), 's-', label=r"Cavity 2 ($|dc_2/d\Phi|$)")
    plt.yscale("log")
    plt.xlabel("Drive Detuning (MHz)")
    plt.ylabel(r"Absolute Quasi-energy Derivative (GHz/$\Phi_0$)")
    plt.title("Cavity Quasi-Energy Derivatives vs Drive Detuning")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
