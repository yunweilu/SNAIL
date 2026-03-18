import argparse
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from qutip import matrix_histogram

import sys
sys.path.insert(0, "/home/yunwei/SNAIL/full_simulation copy/dual_rail/two_dual_rails")
from importlib.machinery import SourceFileLoader
twodualrail = SourceFileLoader("2dualrail", "/home/yunwei/SNAIL/full_simulation copy/dual_rail/two_dual_rails/2dualrail.py").load_module()


def basis_index(t, c2, c1, c3, c4, trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    return t * (n_c2 * n_c1 * n_c3 * n_c4) + c2 * (n_c1 * n_c3 * n_c4) + c1 * (n_c3 * n_c4) + c3 * n_c4 + c4


def build_initial_state(trunc_dim):
    n_t, n_c2, n_c1 = trunc_dim
    n_c3 = 2
    n_c4 = 2
    ket00 = qt.tensor(
        qt.basis(n_t, 0),
        qt.basis(n_c2, 1),
        qt.basis(n_c1, 1),
        qt.basis(n_c3, 0),
        qt.basis(n_c4, 0),
    )
    ket11 = qt.tensor(
        qt.basis(n_t, 0),
        qt.basis(n_c2, 0),
        qt.basis(n_c1, 0),
        qt.basis(n_c3, 1),
        qt.basis(n_c4, 1),
    )
    return (ket00 + ket11).unit()


def build_dressed_subspace_from_h0(H0_total, trunc_dim):
    H_arr = np.asarray(H0_total.full(), dtype=complex)
    _, U = np.linalg.eigh(H_arr)

    n_t, n_c2, n_c1 = trunc_dim
    label_to_index = {
        "00_LL": basis_index(0, 1, 1, 0, 0, trunc_dim),
        "01_LL": basis_index(0, 0, 1, 0, 1, trunc_dim),
        "10_LL": basis_index(0, 1, 0, 1, 0, trunc_dim),
        "11_LL": basis_index(0, 0, 0, 1, 1, trunc_dim),
    }

    dims_ket = [[n_t, n_c2, n_c1, 2, 2], [1, 1, 1, 1, 1]]
    dressed_kets = {}
    dressed_projectors = {}
    used_cols = set()
    for lbl, bare_idx in label_to_index.items():
        overlaps = np.abs(U[bare_idx, :]) ** 2
        order = np.argsort(overlaps)[::-1]
        col = None
        for cand in order:
            if int(cand) not in used_cols:
                col = int(cand)
                break
        if col is None:
            continue
        used_cols.add(col)
        ket = qt.Qobj(U[:, col], dims=dims_ket)
        dressed_kets[lbl] = ket
        dressed_projectors[lbl] = ket * ket.dag()

    return dressed_kets, dressed_projectors


def logical_density_from_dressed_kets(rho_full, dressed_kets):
    labels = ["00_LL", "01_LL", "10_LL", "11_LL"]
    rho_logical_arr = np.zeros((4, 4), dtype=complex)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            if li in dressed_kets and lj in dressed_kets:
                val = dressed_kets[li].dag() * (rho_full * dressed_kets[lj])
                if isinstance(val, qt.Qobj):
                    rho_logical_arr[i, j] = val.full()[0, 0]
                else:
                    rho_logical_arr[i, j] = complex(val)

    rho_logical = qt.Qobj(rho_logical_arr, dims=[[4], [4]])
    if np.abs(rho_logical.tr()) > 0:
        rho_logical = rho_logical / rho_logical.tr()
    return rho_logical


def plot_single_logical_matrix(rho_logical, title, filename):
    labels = ["|00>", "|01>", "|10>", "|11>"]
    rho_abs = qt.Qobj(np.abs(rho_logical.full()), dims=rho_logical.dims)
    fig = plt.figure(figsize=(5.0, 4.6))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    matrix_histogram(
        rho_abs,
        labels,
        labels,
        fig=fig,
        ax=ax,
        limits=[0.0, 0.5],
        bar_style="abs",
        color_style="abs",
        color_limits=[0.0, 0.5],
        cmap=mpl.cm.jet,
        colorbar=False,
    )
    ax.set_title(title)
    ax.view_init(azim=-55, elev=45)
    ax.set_zlim(0.0, 0.5)

    shared_norm = mpl.colors.Normalize(vmin=0.0, vmax=0.5)
    shared_mappable = mpl.cm.ScalarMappable(norm=shared_norm, cmap=mpl.cm.jet)
    shared_mappable.set_array([])
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.18, 0.03, 0.64])
    cbar = fig.colorbar(shared_mappable, cax=cax)
    cbar.set_label("Matrix element magnitude")

    plot_out = Path(filename).resolve()
    fig.savefig(plot_out, dpi=200, bbox_inches="tight")
    print(f"Saved driven initial logical matrix to: {plot_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot driven-case initial logical density matrix (dressed-state average, no simulation)."
    )
    parser.add_argument(
        "--A-over-pi",
        type=float,
        default=10e-3,
        help="Drive amplitude factor with A = (A-over-pi) * 2*pi.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tomo_initial_driven_only.png",
        help="Output figure path.",
    )
    args = parser.parse_args()

    driven_A = args.A_over_pi * 2 * np.pi
    ops_driven = twodualrail.build_total_operators(A=driven_A)
    trunc_dim = ops_driven.get("trunc_dim", [3, 2, 2])
    dressed_kets, dressed_projectors = build_dressed_subspace_from_h0(
        ops_driven["H0_total"], trunc_dim
    )

    if "00_LL" in dressed_kets and "11_LL" in dressed_kets:
        psi0 = (dressed_kets["00_LL"] + dressed_kets["11_LL"]).unit()
    else:
        # Fallback if dressed matching fails for any reason.
        psi0 = build_initial_state(trunc_dim)
    rho0 = qt.ket2dm(psi0)
    rho0_logical = logical_density_from_dressed_kets(rho0, dressed_kets)

    plot_single_logical_matrix(
        rho0_logical,
        "Driven Initial (Dressed Logical Matrix)",
        args.output,
    )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
