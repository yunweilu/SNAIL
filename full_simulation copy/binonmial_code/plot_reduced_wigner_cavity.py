import argparse
import pickle
from pathlib import Path
from datetime import datetime
from importlib.machinery import SourceFileLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from matplotlib.colors import LinearSegmentedColormap

from matrix_elements_playground import build_matrix_elements


# Diverging map with explicit white center so W=0 is white.
WIGNER_CMAP = LinearSegmentedColormap.from_list(
    "wigner_blue_white_red",
    ["#2166ac", "#ffffff", "#b2182b"],
    N=256,
)


def _style_setup():
    root = Path(__file__).resolve().parents[3]
    candidate_paths = [
        root / "SNAIL_new" / "plot_instruction",
        root / "plot prompt",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            style_mod = SourceFileLoader("plot_prompt_wigner_mod", str(path)).load_module()
            mpl.rcParams.update(style_mod.normal_plot)
            return style_mod
        except SyntaxError:
            continue
    raise RuntimeError(
        f"Could not load plot style module from: {[str(p) for p in candidate_paths]}"
    )


def load_reduced_cavity_data(data_path, case="undriven"):
    with Path(data_path).open("rb") as f:
        payload = pickle.load(f)

    # Backward compatibility:
    # old format -> top-level time_points, avg_rho_cav
    # new format -> payload["undriven"/"driven"]["time_points"/"avg_rho_cav"]
    if "time_points" in payload and "avg_rho_cav" in payload:
        time_points = np.asarray(payload["time_points"])
        avg_rho_cav = np.asarray(payload["avg_rho_cav"])
    else:
        if case not in payload:
            raise KeyError(f"Case '{case}' not found. Available keys: {list(payload.keys())}")
        time_points = np.asarray(payload[case]["time_points"])
        avg_rho_cav = np.asarray(payload[case]["avg_rho_cav"])

    if avg_rho_cav.ndim != 3 or avg_rho_cav.shape[1:] != (5, 5):
        raise ValueError("Expected avg_rho_cav with shape (n_t, 5, 5).")
    return time_points, avg_rho_cav


def remove_self_kerr_phase(rho_t, t, H_kerr):
    U_t = (1j * H_kerr * t).expm()
    return U_t * rho_t * U_t.dag()


def build_kerr_hamiltonian(self_kerr, n_cavity=5):
    n_op = qt.num(n_cavity)
    return 0.5 * float(self_kerr) * n_op * (n_op - qt.qeye(n_cavity))


def apply_self_kerr_phase_removal(avg_rho_cav, time_points, self_kerr):
    H_kerr = build_kerr_hamiltonian(self_kerr, n_cavity=avg_rho_cav.shape[1])
    corrected = np.empty_like(avg_rho_cav, dtype=complex)
    for idx, t in enumerate(time_points):
        rho = qt.Qobj(avg_rho_cav[idx], dims=[[avg_rho_cav.shape[1]], [avg_rho_cav.shape[1]]])
        corrected[idx] = remove_self_kerr_phase(rho, float(t), H_kerr).full()
    return corrected


def estimate_best_snapshot_phase(rho_cav, min_coherence=1e-10, grid_points=721):
    n_cavity = rho_cav.shape[0]
    g = 0.5 * np.arange(n_cavity, dtype=float) * (np.arange(n_cavity, dtype=float) - 1.0)
    phases = []
    deltas = []
    weights = []
    for i in range(n_cavity):
        for j in range(i + 1, n_cavity):
            delta = g[i] - g[j]
            if np.abs(delta) < 1e-14:
                continue
            cij = rho_cav[i, j]
            amp = np.abs(cij)
            if amp <= min_coherence:
                continue
            phases.append(np.angle(cij))
            deltas.append(delta)
            weights.append(float(amp * amp))

    if not phases:
        return 0.0

    phases = np.asarray(phases, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    phi_grid = np.linspace(-np.pi, np.pi, int(grid_points), endpoint=True)
    score = np.zeros_like(phi_grid)
    for k, phi in enumerate(phi_grid):
        score[k] = np.sum(weights * np.cos(phases + phi * deltas))
    return float(phi_grid[int(np.argmax(score))])


def apply_snapshot_phase_removal(avg_rho_cav, min_coherence=1e-10, grid_points=721):
    n_cavity = avg_rho_cav.shape[1]
    n_op = qt.num(n_cavity)
    G = 0.5 * n_op * (n_op - qt.qeye(n_cavity))
    corrected = np.empty_like(avg_rho_cav, dtype=complex)
    phase_list = []
    for idx in range(avg_rho_cav.shape[0]):
        rho_arr = np.asarray(avg_rho_cav[idx], dtype=complex)
        phi = estimate_best_snapshot_phase(rho_arr, min_coherence=min_coherence, grid_points=grid_points)
        U = (1j * G * phi).expm()
        rho = qt.Qobj(rho_arr, dims=[[n_cavity], [n_cavity]])
        corrected[idx] = (U * rho * U.dag()).full()
        phase_list.append(phi)
    return corrected, np.asarray(phase_list, dtype=float)


def apply_driven_phase_align_to_t0(avg_rho_cav, grid_points=721):
    """
    For each time slice, optimize Kerr-phase phi to maximize fidelity with rho(t=0).
    """
    n_cavity = avg_rho_cav.shape[1]
    n_op = qt.num(n_cavity)
    G = 0.5 * n_op * (n_op - qt.qeye(n_cavity))
    phi_grid = np.linspace(-np.pi, np.pi, int(grid_points), endpoint=True)

    corrected = np.empty_like(avg_rho_cav, dtype=complex)
    phase_list = []
    fidelity_list = []

    rho_ref = qt.Qobj(np.asarray(avg_rho_cav[0], dtype=complex), dims=[[n_cavity], [n_cavity]])
    corrected[0] = rho_ref.full()
    phase_list.append(0.0)
    fidelity_list.append(1.0)

    for idx in range(1, avg_rho_cav.shape[0]):
        rho = qt.Qobj(np.asarray(avg_rho_cav[idx], dtype=complex), dims=[[n_cavity], [n_cavity]])
        best_phi = 0.0
        best_fid = -1.0
        best_rho = rho
        for phi in phi_grid:
            U = (1j * G * float(phi)).expm()
            rho_corr = U * rho * U.dag()
            fid = float(qt.metrics.fidelity(rho_ref, rho_corr))
            if fid > best_fid:
                best_fid = fid
                best_phi = float(phi)
                best_rho = rho_corr
        corrected[idx] = best_rho.full()
        phase_list.append(best_phi)
        fidelity_list.append(best_fid)

    return corrected, np.asarray(phase_list, dtype=float), np.asarray(fidelity_list, dtype=float)


def fit_self_kerr_from_density_matrices(avg_rho_cav, time_points, min_coherence=1e-10):
    """
    Fit effective Kerr K from phase evolution of off-diagonal elements rho_ij(t):
      d/dt arg[rho_ij] ~ -K * (i(i-1)-j(j-1))/2
    """
    time_points = np.asarray(time_points, dtype=float)
    avg_rho_cav = np.asarray(avg_rho_cav, dtype=complex)
    if avg_rho_cav.ndim != 3 or avg_rho_cav.shape[0] < 3:
        return float("nan")

    n_cavity = avg_rho_cav.shape[1]
    k_list = []
    w_list = []
    for i in range(n_cavity):
        for j in range(i + 1, n_cavity):
            delta = 0.5 * (i * (i - 1) - j * (j - 1))
            if np.abs(delta) < 1e-14:
                continue
            coh = avg_rho_cav[:, i, j]
            amp = np.abs(coh)
            mask = amp > min_coherence
            if np.count_nonzero(mask) < 3:
                continue

            t = time_points[mask]
            phi = np.unwrap(np.angle(coh[mask]))
            w = np.maximum(amp[mask] ** 2, 1e-16)

            # Weighted least-squares fit: phi(t) = a + b*t
            sw = np.sum(w)
            st = np.sum(w * t)
            sp = np.sum(w * phi)
            stt = np.sum(w * t * t)
            stp = np.sum(w * t * phi)
            denom = sw * stt - st * st
            if np.abs(denom) < 1e-20:
                continue
            slope = (sw * stp - st * sp) / denom
            k_ij = -slope / delta

            k_list.append(k_ij)
            w_list.append(float(np.mean(w) * np.abs(delta)))

    if not k_list:
        return float("nan")
    return float(np.average(np.asarray(k_list), weights=np.asarray(w_list)))


def get_case_self_kerr(data_path, case):
    with Path(data_path).open("rb") as f:
        payload = pickle.load(f)
    if case not in payload:
        return 0.0
    params = payload.get("params", {})
    A_case = payload[case].get("A", 0.0)
    n_transmon = int(params.get("n_transmon", 3))
    n_cavity = int(params.get("n_cavity", 5))
    elem = build_matrix_elements(A_case, n_transmon=n_transmon, n_cavity=n_cavity)
    return float(elem["self_kerr_h0_rot"])


def load_reduced_cavity_state(
    data_path,
    case="undriven",
    index=None,
    time_ns=None,
    fit_driven_self_kerr=False,
    per_snapshot_phase_correction=False,
):
    time_points, avg_rho_cav = load_reduced_cavity_data(data_path, case=case)
    if case == "driven":
        avg_rho_cav, phases, fids = apply_driven_phase_align_to_t0(avg_rho_cav)
        print(
            "Driven per-slice phase alignment to t=0: "
            f"median(phi)={np.median(phases):.6f} rad, "
            f"min fidelity={np.min(fids):.6f}, median fidelity={np.median(fids):.6f}"
        )
    elif per_snapshot_phase_correction:
        avg_rho_cav, phases = apply_snapshot_phase_removal(avg_rho_cav)
        print(
            f"Applied per-snapshot phase correction ({case}): "
            f"median={np.median(phases):.6f} rad"
        )
    else:
        self_kerr = get_case_self_kerr(data_path, case=case)
        if fit_driven_self_kerr and case == "driven":
            k_fit = fit_self_kerr_from_density_matrices(avg_rho_cav, time_points)
            if np.isfinite(k_fit):
                print(f"Driven self-Kerr: model={self_kerr:.6e}, fitted={k_fit:.6e}")
                self_kerr = float(k_fit)
            else:
                print(f"Driven self-Kerr fit failed; using model value {self_kerr:.6e}")
        avg_rho_cav = apply_self_kerr_phase_removal(avg_rho_cav, time_points, self_kerr)

    if time_ns is not None:
        idx = int(np.argmin(np.abs(time_points - time_ns)))
    elif index is not None:
        idx = int(index)
    else:
        idx = -1  # default: final time

    if idx < -len(time_points) or idx >= len(time_points):
        raise IndexError(f"Index {idx} out of range for {len(time_points)} time points.")

    rho_cav = qt.Qobj(avg_rho_cav[idx], dims=[[5], [5]])
    return rho_cav, float(time_points[idx]), idx


def _apply_case_correction(data_path, case, time_points, avg_rho_cav, fit_driven_self_kerr, per_snapshot_phase_correction):
    if case == "driven":
        corrected, phases, fids = apply_driven_phase_align_to_t0(avg_rho_cav)
        print(
            "Driven per-slice phase alignment to t=0: "
            f"median(phi)={np.median(phases):.6f} rad, "
            f"min fidelity={np.min(fids):.6f}, median fidelity={np.median(fids):.6f}"
        )
        return corrected

    if per_snapshot_phase_correction:
        corrected, phases = apply_snapshot_phase_removal(avg_rho_cav)
        print(
            f"Applied per-snapshot phase correction ({case}): "
            f"median={np.median(phases):.6f} rad"
        )
        return corrected

    k_case = get_case_self_kerr(data_path, case=case)
    if fit_driven_self_kerr and case == "driven":
        k_fit = fit_self_kerr_from_density_matrices(avg_rho_cav, time_points)
        if np.isfinite(k_fit):
            print(f"Driven self-Kerr: model={k_case:.6e}, fitted={k_fit:.6e}")
            k_case = float(k_fit)
        else:
            print(f"Driven self-Kerr fit failed; using model value {k_case:.6e}")
    print(f"Using self-Kerr correction ({case}): K={k_case:.6e}")
    return apply_self_kerr_phase_removal(avg_rho_cav, time_points, k_case)


def report_fidelity_t0_to_target_us(time_points, avg_rho_cav, case_label, target_us=300.0):
    """Print fidelity between t=0 and t≈target_us for a given case."""
    if len(time_points) == 0:
        print(f"{case_label}: no time points available for fidelity report.")
        return

    target_ns = float(target_us) * 1e3
    idx_target = int(np.argmin(np.abs(time_points - target_ns)))

    rho_0 = qt.Qobj(np.asarray(avg_rho_cav[0], dtype=complex), dims=[[5], [5]])
    rho_t = qt.Qobj(np.asarray(avg_rho_cav[idx_target], dtype=complex), dims=[[5], [5]])
    fid = float(qt.metrics.fidelity(rho_0, rho_t))

    print(
        f"{case_label}: F[t=0, t≈{target_us:.0f} us] = {fid:.6f} "
        f"(nearest t={time_points[idx_target] / 1e3:.2f} us, idx={idx_target})"
    )


def plot_wigner(rho_cav, t_value, xlim=6.0, grid=201, save_path=None, dpi=200):
    x = np.linspace(-xlim, xlim, grid)
    W = qt.wigner(rho_cav, x, x)

    fig, ax = plt.subplots(figsize=(6, 5))
    levels = np.linspace(W.min(), W.max(), 120)
    cf = ax.contourf(x, x, W, levels=levels, cmap="RdBu_r")
    ax.contour(x, x, W, levels=[0], colors="k", linewidths=0.8)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title(f"Cavity Wigner function (transmon traced out), t={t_value:.2f} ns")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, label="W(q,p)")
    plt.tight_layout()
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {out.resolve()}")
    plt.show()


def plot_wigner_10points(avg_rho_cav, time_points, xlim=6.0, grid=201, save_path=None, dpi=200):
    n_t = len(time_points)
    if n_t == 0:
        raise ValueError("No time points found in input data.")

    n_plot = min(10, n_t)
    if n_t <= 10:
        plot_indices = np.arange(n_t, dtype=int)
    else:
        plot_indices = np.linspace(0, n_t - 1, n_plot, dtype=int)

    x = np.linspace(-xlim, xlim, grid)
    w_list = []
    for idx in plot_indices:
        rho_cav = qt.Qobj(avg_rho_cav[idx], dims=[[5], [5]])
        w_list.append(qt.wigner(rho_cav, x, x))

    wmin = min(np.min(w) for w in w_list)
    wmax = max(np.max(w) for w in w_list)
    levels = np.linspace(wmin, wmax, 120)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    axes = axes.ravel()
    cf = None
    for ax, idx, W in zip(axes, plot_indices, w_list):
        cf = ax.contourf(x, x, W, levels=levels, cmap=WIGNER_CMAP)
        ax.contour(x, x, W, levels=[0], colors="k", linewidths=0.6)
        ax.set_title(f"t={time_points[idx]:.2f} ns")
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        ax.set_aspect("equal")

    # In case n_t < 10, hide unused axes.
    for ax in axes[len(plot_indices):]:
        ax.axis("off")

    if cf is not None:
        fig.colorbar(cf, ax=axes.tolist(), label="W(q,p)", shrink=0.9)
    fig.suptitle("Cavity Wigner function (transmon traced out) at 10 time points")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {out.resolve()}")
    plt.show()


def plot_wigner_three_snapshots(
    undriven_data,
    driven_data,
    style_mod,
    xlim=6.0,
    grid=201,
    save_path=None,
    dpi=200,
):
    """
    Plot three requested panels:
      1) undriven at t=0
      2) undriven at final t
      3) driven at final t
    """
    t_u, rho_u = undriven_data
    t_d, rho_d = driven_data
    if len(t_u) == 0 or len(t_d) == 0:
        raise ValueError("Time points are empty for undriven or driven data.")

    idx_u0 = 0
    idx_uf = len(t_u) - 1
    idx_df = len(t_d) - 1

    snapshots = [
        ("Undriven", float(t_u[idx_u0]), qt.Qobj(rho_u[idx_u0], dims=[[5], [5]])),
        ("Undriven", float(t_u[idx_uf]), qt.Qobj(rho_u[idx_uf], dims=[[5], [5]])),
        ("Driven", float(t_d[idx_df]), qt.Qobj(rho_d[idx_df], dims=[[5], [5]])),
    ]

    x = np.linspace(-xlim, xlim, grid)
    w_list = [qt.wigner(rho, x, x) for _, _, rho in snapshots]
    # Use one shared symmetric color range across all three panels.
    wmin = min(np.min(w) for w in w_list)
    wmax = max(np.max(w) for w in w_list)
    vmax = max(abs(wmin), abs(wmax))
    levels = np.linspace(-vmax, vmax, 121)

    base_w, base_h = mpl.rcParams["figure.figsize"]
    fig, axes = plt.subplots(1, 3, figsize=(base_w * 2.9, base_h), constrained_layout=True)
    cf = None
    for panel_idx, (ax, (label, tval_ns, _rho), W) in enumerate(zip(axes, snapshots, w_list)):
        tval_us = tval_ns / 1e3
        cf = ax.contourf(x, x, W, levels=levels, cmap="RdBu_r")
        if panel_idx == 0:
            ax.set_title(f"$t={tval_us:.0f}\\,\\mu s$")
            ax.set_ylabel("p")
        else:
            ax.set_title(f"{label}, $t={tval_us:.0f}\\,\\mu s$")
            ax.set_ylabel("")
        ax.set_xlabel("q")
        ax.set_aspect("equal")
        style_mod.add_axis_arrows(ax)

    if cf is not None:
        fig.colorbar(cf, ax=axes.ravel().tolist(), label="W(q,p)", shrink=0.92, format="%.1f")

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to: {out.resolve()}")
    plt.show()


def main():
    style_mod = _style_setup()
    parser = argparse.ArgumentParser(
        description="Plot cavity reduced Wigner function from averaged density-matrix data."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="avg_density_matrix_over_time_undriven_driven.pkl",
        help="Pickle file generated by average_density_matrix_simulation.py",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["undriven", "driven", "both"],
        default="both",
        help="Which simulation case to plot for new-format files.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Time index to plot (default: final time).",
    )
    parser.add_argument(
        "--time-ns",
        type=float,
        default=None,
        help="Physical time in ns (plots nearest point). Overrides --index.",
    )
    parser.add_argument("--xlim", type=float, default=6.0, help="Wigner axis limit.")
    parser.add_argument("--grid", type=int, default=201, help="Wigner grid size.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Plot a single time point (use --index or --time-ns). Default is 10 points in one PNG.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional path to save the figure. Default: binonmial_code folder with auto filename.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure.")
    parser.add_argument(
        "--fit-driven-self-kerr",
        action="store_true",
        help="Fit effective driven self-Kerr from density matrices before phase removal.",
    )
    parser.add_argument(
        "--per-snapshot-phase-correction",
        action="store_true",
        help="For each time slice, fit and remove a Kerr-like phase independently.",
    )
    args = parser.parse_args()

    if args.save_path is None:
        if args.single:
            mode = "single"
        elif args.case == "both":
            mode = "3snapshots"
        else:
            mode = "10points"
        default_name = f"cavity_wigner_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        save_path = Path(__file__).resolve().parent / default_name
    else:
        save_path = Path(args.save_path)

    if args.single and args.case == "both":
        raise ValueError("--single requires --case undriven or --case driven.")

    if args.single:
        rho_cav, t_value, idx = load_reduced_cavity_state(
            data_path=args.data,
            case=args.case,
            index=args.index,
            time_ns=args.time_ns,
            fit_driven_self_kerr=args.fit_driven_self_kerr,
            per_snapshot_phase_correction=args.per_snapshot_phase_correction,
        )
        print(
            f"Plotting single ({args.case}) index={idx}, time={t_value:.4f} ns "
            f"from {Path(args.data).resolve()}"
        )
        plot_wigner(
            rho_cav,
            t_value=t_value,
            xlim=args.xlim,
            grid=args.grid,
            save_path=save_path,
            dpi=args.dpi,
        )
    else:
        if args.case == "both":
            undriven_data = load_reduced_cavity_data(args.data, case="undriven")
            driven_data = load_reduced_cavity_data(args.data, case="driven")
            undriven_data = (
                undriven_data[0],
                _apply_case_correction(
                    args.data,
                    "undriven",
                    undriven_data[0],
                    undriven_data[1],
                    fit_driven_self_kerr=args.fit_driven_self_kerr,
                    per_snapshot_phase_correction=args.per_snapshot_phase_correction,
                ),
            )
            driven_data = (
                driven_data[0],
                _apply_case_correction(
                    args.data,
                    "driven",
                    driven_data[0],
                    driven_data[1],
                    fit_driven_self_kerr=args.fit_driven_self_kerr,
                    per_snapshot_phase_correction=args.per_snapshot_phase_correction,
                ),
            )

            # Fidelity report at t=0 vs t≈300 us for both cases.
            report_fidelity_t0_to_target_us(
                undriven_data[0], undriven_data[1], case_label="Undriven", target_us=300.0
            )
            report_fidelity_t0_to_target_us(
                driven_data[0], driven_data[1], case_label="Driven", target_us=300.0
            )

            print(f"Plotting three requested snapshots (both cases) from {Path(args.data).resolve()}")
            plot_wigner_three_snapshots(
                undriven_data=undriven_data,
                driven_data=driven_data,
                style_mod=style_mod,
                xlim=args.xlim,
                grid=args.grid,
                save_path=save_path,
                dpi=args.dpi,
            )
        else:
            time_points, avg_rho_cav = load_reduced_cavity_data(args.data, case=args.case)
            avg_rho_cav = _apply_case_correction(
                args.data,
                args.case,
                time_points,
                avg_rho_cav,
                fit_driven_self_kerr=args.fit_driven_self_kerr,
                per_snapshot_phase_correction=args.per_snapshot_phase_correction,
            )
            print(f"Plotting 10-point panel ({args.case}) from {Path(args.data).resolve()}")
            plot_wigner_10points(
                avg_rho_cav=avg_rho_cav,
                time_points=time_points,
                xlim=args.xlim,
                grid=args.grid,
                save_path=save_path,
                dpi=args.dpi,
            )


if __name__ == "__main__":
    main()
