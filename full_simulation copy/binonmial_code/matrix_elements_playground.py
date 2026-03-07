import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from system import *
from hamiltonian_generator import Hamiltonian


def cavity_self_kerr_from_diag(diag_h, n_cavity):
    """Estimate cavity self-Kerr from |0,0>, |0,1>, |0,2> ladder."""
    if n_cavity < 3 or diag_h.shape[0] < 3:
        return float("nan")
    e0 = np.real(diag_h[0, 0])
    e1 = np.real(diag_h[1, 1])
    e2 = np.real(diag_h[2, 2])
    return float(e2 - 2.0 * e1 + e0)


def build_matrix_elements(A, n_transmon=3, n_cavity=5, phase_02=0.0, phase_04=0.0):
    phi_ex = 0.2
    Ej = 30.19
    Ec = 0.1
    op_dims = [[n_transmon, n_cavity], [n_transmon, n_cavity]]
    tol = 1e-12

    sc_tmp = Hamiltonian(phi_ex, Ej, Ec, [5, 10])
    optimal_omega, _ = sc_tmp.optimal_omegad(A)
    optimal_omega = optimal_omega * 2 * np.pi

    sc = Hamiltonian(phi_ex, Ej, Ec, [n_transmon, n_cavity])
    _, kick_and_sigmax, _ = sc.setup_floquet_system(A, optimal_omega)
    total_dim = n_transmon * n_cavity

    a_s_dressed = np.kron(sc.annihilation(n_transmon), np.eye(n_cavity))
    a_c_dressed = np.kron(np.eye(n_transmon), sc.annihilation(n_cavity))
    n_s = np.kron(np.diag(np.arange(n_transmon)), np.eye(n_cavity))
    n_c = np.kron(np.eye(n_transmon), np.diag(np.arange(n_cavity)))

    sds_raw = np.asarray(sc.noise, dtype=complex).copy()
    diag_vals = np.diag(sds_raw) - sds_raw[0, 0]
    np.fill_diagonal(sds_raw, diag_vals)
    x_s = a_s_dressed + a_s_dressed.conj().T
    x_mask = np.abs(x_s) > tol
    diag_mask = np.eye(sds_raw.shape[0], dtype=bool)
    sds_keep_mask = diag_mask 
    sds_diag = np.zeros_like(sds_raw, dtype=complex)
    sds_diag[sds_keep_mask] = sds_raw[sds_keep_mask]

    sop_raw = np.asarray(sc.s)
    s_mask = np.abs(a_s_dressed) > tol
    c_mask = np.abs(a_c_dressed) > tol
    sop_selected = np.zeros_like(sop_raw, dtype=complex)
    cop_selected = np.zeros_like(sop_raw, dtype=complex)
    sop_selected[s_mask] = sop_raw[s_mask]
    cop_selected[c_mask] = sop_raw[c_mask]

    H_control_filtered = np.zeros_like(np.asarray(sc.H_control), dtype=complex)
    for i in range(total_dim - n_cavity):
        j = i + n_cavity
        H_control_filtered[i, j] = sc.H_control[i, j]
        H_control_filtered[j, i] = sc.H_control[j, i]
    
    H0_diag = np.diag(np.diag(sc.H) - sc.H[0, 0])
    omegac = H0_diag[1,1]
    H0_rot_raw = H0_diag - optimal_omega * n_s - omegac * n_c + (A / 2.0) * H_control_filtered
    evals_rot, U = np.linalg.eigh(H0_rot_raw)
    evals_rot, U = sort_eigenpairs(evals_rot, U)
    evals_rot = np.real(evals_rot - evals_rot[0])
    H0_rot = np.diag(evals_rot)
    omega_c = float(evals_rot[1]) if len(evals_rot) > 1 else float(np.real(omegac))
    new_c = U.T.conj() @ cop_selected @ U
    new_s = U.T.conj() @ sop_selected @ U
    new_sds = U.T.conj() @ sds_diag @ U
    ket00 = U[:, 0]
    ket02 = U[:, 2]
    ket04 = U[:, 4]
    if np.abs(ket00[0]) > 1e-15:
        ket00 = ket00 / ket00[0]
    if np.abs(ket02[2]) > 1e-15:
        ket02 = ket02 / ket02[2]
    if np.abs(ket04[4]) > 1e-15:
        ket04 = ket04 / ket04[4]
    n00 = np.linalg.norm(ket00)
    n02 = np.linalg.norm(ket02)
    n04 = np.linalg.norm(ket04)
    if n00 > 1e-15:
        ket00 = ket00 / n00
    if n02 > 1e-15:
        ket02 = ket02 / n02
    if n04 > 1e-15:
        ket04 = ket04 / n04
    phase02 = np.exp(1j * phase_02)
    phase04 = np.exp(1j * phase_04)
    ket0_u = np.sqrt(1 / 2) * (ket00 + phase04 * ket04)
    ket1_u = phase02 * ket02
    initial_state_u = np.sqrt(1 / 2) * (ket0_u + ket1_u)

    ket0 = qt.Qobj(kick_and_sigmax(0)[0][:, 0], dims=[[n_transmon, n_cavity], [1, 1]])
    ket2 = qt.Qobj(kick_and_sigmax(0)[0][:, 2], dims=[[n_transmon, n_cavity], [1, 1]])
    ket4 = qt.Qobj(kick_and_sigmax(0)[0][:, 4], dims=[[n_transmon, n_cavity], [1, 1]])
    overlap_00_0 = np.vdot(ket00, ket0.full().ravel())
    overlap_02_2 = np.vdot(ket02, ket2.full().ravel())
    overlap_04_4 = np.vdot(ket04, ket4.full().ravel())
    ket0L = (ket0 + ket4).unit()
    ket1L = ket2
    logical_initial_state = (ket0L + ket1L).unit()


 


    payload = {
        "dims": op_dims,
        "n_transmon": n_transmon,
        "n_cavity": n_cavity,
        "A": A,
        "phase_02": phase_02,
        "phase_04": phase_04,
        "optimal_omega": optimal_omega,
        "omega_c": omega_c,
        "a_s_dressed": a_s_dressed,
        "a_c_dressed": a_c_dressed,
        "sds_diag": sds_diag,
        "sop_raw": sop_raw,
        "sop_selected": sop_selected,
        "cop_selected": cop_selected,
        "H_control_filtered": H_control_filtered,
        "H0_diag": H0_diag,
        "H0_rot": H0_rot,
        "self_kerr_h0_diag": cavity_self_kerr_from_diag(H0_diag, n_cavity),
        "self_kerr_h0_rot": cavity_self_kerr_from_diag(H0_rot, n_cavity),
        "new_c": new_c,
        "new_s": new_s,
        "new_sds": new_sds,
        "initial_state_u": initial_state_u,
        "logical_initial_state": logical_initial_state,
        "overlap_00_0": overlap_00_0,
        "overlap_02_2": overlap_02_2,
        "overlap_04_4": overlap_04_4,
        "s_nonzero": np.argwhere(s_mask),
        "c_nonzero": np.argwhere(c_mask),
        "sop_selected_nonzero": np.argwhere(np.abs(sop_selected) > tol),
        "cop_selected_nonzero": np.argwhere(np.abs(cop_selected) > tol),
    }
    return payload


def format_nm(index, n_cavity):
    n = int(index) // n_cavity
    m = int(index) % n_cavity
    return f"{n}{m}"


def print_nonzero(name, mat, n_cavity, max_entries=None, tol=1e-12):
    idx = np.argwhere(np.abs(mat) > tol)
    print(f"\n{name}: {len(idx)} nonzero entries")
    if max_entries is not None:
        idx = idx[:max_entries]
    for i, j in idx:
        val = mat[i, j]
        print(
            f"  ({format_nm(i, n_cavity)}, {format_nm(j, n_cavity)}) -> "
            f"{val}    |val|={np.abs(val):.6e}"
        )


def plot_initial_state_wigner(
    initial_state, n_transmon, n_cavity, title, xlim=6.0, grid=201, save_path=None, dpi=200
):
    if isinstance(initial_state, qt.Qobj):
        psi = initial_state
    else:
        psi = qt.Qobj(initial_state, dims=[[n_transmon, n_cavity], [1, 1]])
    rho_cavity = qt.ket2dm(psi).ptrace(1)
    x = np.linspace(-xlim, xlim, grid)
    W = qt.wigner(rho_cavity, x, x)

    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(W.min(), W.max(), 120)
    cf = ax.contourf(x, x, W, levels=levels, cmap="RdBu_r")
    ax.contour(x, x, W, levels=[0], colors="k", linewidths=0.8)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, label="W(q,p)")
    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved Wigner plot to: {out.resolve()}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Play with dressed matrix elements only.")
    parser.add_argument("--A-over-pi", type=float, default=10e-3, help="Drive amplitude as A/pi")
    parser.add_argument("--phase-02", type=float, default=0.0, help="Relative phase (radians) on ket02.")
    parser.add_argument("--phase-04", type=float, default=0.0, help="Relative phase (radians) on ket04.")
    parser.add_argument("--n-transmon", type=int, default=3, help="Transmon Hilbert-space truncation.")
    parser.add_argument("--n-cavity", type=int, default=5, help="Cavity Hilbert-space truncation.")
    parser.add_argument("--max-print", type=int, default=80, help="Max entries to print per matrix")
    parser.add_argument("--save", type=str, default="", help="Optional .npz output path")
    parser.add_argument("--wigner-xlim", type=float, default=6.0, help="Wigner axis limit.")
    parser.add_argument("--wigner-grid", type=int, default=201, help="Wigner grid size.")
    parser.add_argument("--wigner-dpi", type=int, default=200, help="DPI for Wigner image.")
    parser.add_argument(
        "--wigner-save-path",
        type=str,
        default="",
        help="Optional save path for initial-state Wigner PNG (default: auto filename).",
    )
    args = parser.parse_args()

    A = args.A_over_pi * np.pi
    data = build_matrix_elements(
        A,
        n_transmon=args.n_transmon,
        n_cavity=args.n_cavity,
        phase_02=args.phase_02,
        phase_04=args.phase_04,
    )

    print(f"A = {A:.6e}")
    print(f"phase_02 = {data['phase_02']:.6f} rad")
    print(f"phase_04 = {data['phase_04']:.6f} rad")
    print(f"optimal_omega = {data['optimal_omega']:.6e}")
    print(f"omega_c (from H0) = {data['omega_c']:.6e}")
    print(f"dims = {data['dims']}")

    undriven_data = build_matrix_elements(
        0.0,
        n_transmon=args.n_transmon,
        n_cavity=args.n_cavity,
        phase_02=args.phase_02,
        phase_04=args.phase_04,
    )
    print(f"self-Kerr (undriven, A=0, from H0_rot) = {undriven_data['self_kerr_h0_rot']:.6e}")
    print(f"self-Kerr (driven, A={A:.6e}, from H0_rot) = {data['self_kerr_h0_rot']:.6e}")

    n_cavity = int(data["n_cavity"])

    print_nonzero("H0_diag", data["H0_diag"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("sds", data["sds_diag"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("sop_selected", data["sop_selected"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("cop_selected", data["cop_selected"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("H_control_filtered", data["H_control_filtered"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("new_s", data["new_s"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("new_c", data["new_c"], n_cavity=n_cavity, max_entries=args.max_print)
    print_nonzero("new_sds", data["new_sds"], n_cavity=n_cavity, max_entries=args.max_print)
    print(f"\n<ket00|ket0> = {data['overlap_00_0']}  |.|^2 = {np.abs(data['overlap_00_0'])**2:.12f}")
    print(f"<ket02|ket2> = {data['overlap_02_2']}  |.|^2 = {np.abs(data['overlap_02_2'])**2:.12f}")
    print(f"<ket04|ket4> = {data['overlap_04_4']}  |.|^2 = {np.abs(data['overlap_04_4'])**2:.12f}")

    if args.wigner_save_path:
        wigner_save_path = args.wigner_save_path
    else:
        wigner_save_path = (
            Path(__file__).resolve().parent
            / f"initial_state_wigner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
    plot_initial_state_wigner(
        initial_state=data["initial_state_u"],
        n_transmon=int(data["n_transmon"]),
        n_cavity=int(data["n_cavity"]),
        title="U-basis initial-state cavity Wigner function",
        xlim=args.wigner_xlim,
        grid=args.wigner_grid,
        save_path=str(Path(wigner_save_path).with_name(Path(wigner_save_path).stem + "_u.png")),
        dpi=args.wigner_dpi,
    )
    plot_initial_state_wigner(
        initial_state=data["logical_initial_state"],
        n_transmon=int(data["n_transmon"]),
        n_cavity=int(data["n_cavity"]),
        title="Logical (|0_L>+|1_L>)/sqrt(2) cavity Wigner function",
        xlim=args.wigner_xlim,
        grid=args.wigner_grid,
        save_path=str(Path(wigner_save_path).with_name(Path(wigner_save_path).stem + "_logical.png")),
        dpi=args.wigner_dpi,
    )

    if args.save:
        out = Path(args.save)
        np.savez(
            out,
            A=data["A"],
            phase_02=data["phase_02"],
            phase_04=data["phase_04"],
            optimal_omega=data["optimal_omega"],
            omega_c=data["omega_c"],
            n_transmon=data["n_transmon"],
            n_cavity=data["n_cavity"],
            a_s_dressed=data["a_s_dressed"],
            a_c_dressed=data["a_c_dressed"],
            sds_diag=data["sds_diag"],
            sop_raw=data["sop_raw"],
            sop_selected=data["sop_selected"],
            cop_selected=data["cop_selected"],
            H_control_filtered=data["H_control_filtered"],
            H0_diag=data["H0_diag"],
            H0_rot=data["H0_rot"],
            new_s=data["new_s"],
            new_c=data["new_c"],
            new_sds=data["new_sds"],
            initial_state_u=data["initial_state_u"],
            logical_initial_state=data["logical_initial_state"].full(),
            overlap_00_0=data["overlap_00_0"],
            overlap_02_2=data["overlap_02_2"],
            overlap_04_4=data["overlap_04_4"],
            s_nonzero=data["s_nonzero"],
            c_nonzero=data["c_nonzero"],
            sop_selected_nonzero=data["sop_selected_nonzero"],
            cop_selected_nonzero=data["cop_selected_nonzero"],
        )
        print(f"\nSaved matrix payload to: {out.resolve()}")


if __name__ == "__main__":
    main()
