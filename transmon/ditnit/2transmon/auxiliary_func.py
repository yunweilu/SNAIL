from qutip.solver.floquet import _floquet_delta_tensor, _floquet_X_matrices, _floquet_gamma_matrices, _floquet_A_matrix, FloquetBasis, _floquet_dephase_matrix

def obtain_Aw(H, c_ops, spectra_cb, T=0, w_th=0.0, kmax=5, nT=100):
    """
    Construct a tensor that represents the master equation in the floquet
    basis.

    Simplest RWA approximation [Grifoni et al, Phys.Rep. 304 229 (1998)]

    Parameters
    ----------
    H : :obj:`.QobjEvo`, :obj:`.FloquetBasis`
        Periodic Hamiltonian a floquet basis system.

    T : float, optional
        The period of the time-dependence of the hamiltonian. Optional if ``H``
        is a ``FloquetBasis`` object.

    c_ops : list of :class:`.Qobj`
        list of collapse operators.

    spectra_cb : list callback functions
        List of callback functions that compute the noise power spectrum as
        a function of frequency for the collapse operators in `c_ops`.

    w_th : float, default: 0.0
        The temperature in units of frequency.

    kmax : int, default: 5
        The truncation of the number of sidebands (default 5).

    nT : int, default: 100
        The number of integration steps (for calculating X) within one period.

    Returns
    -------
    output : array
        The Floquet-Markov master equation tensor `R`.
    """
    if isinstance(H, FloquetBasis):
        floquet_basis = H
        T = H.T
    else:
        floquet_basis = FloquetBasis(H, T)
    energy = floquet_basis.e_quasi
    delta = _floquet_delta_tensor(energy, kmax, T)
    x = _floquet_X_matrices(floquet_basis, c_ops, kmax, nT)
    gamma = _floquet_gamma_matrices(x, delta, spectra_cb)
    a = _floquet_A_matrix(delta, gamma, w_th)
    w_dephase = _floquet_dephase_matrix(x, spectra_cb, T)
    return a, w_dephase