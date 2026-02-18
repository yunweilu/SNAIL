import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scqubits as scqubits
from scipy.optimize import minimize
import scipy as sci
class Hamiltonian:
    def __init__(self, phi_ex, Ej, Ec, original_dim, trunc_dim, omega_c1, omega_c2, N=1, beta=1):
        """
        phi_ex: External flux in units of flux quanta
        Ej: Josephson energy
        Ec: Charging energy
        original_dim: dimension of the full Hilbert space
        trunc_dim: dimension of the truncated Hilbert space
        omega_c1, omega_c2: frequencies of the two modes
        N: number of large junctions (default 1)
        beta: ratio of junction energies (default 1)
        """

        self.phi_ex = phi_ex
        self.Ej = Ej * 2 * np.pi
        self.Ec = Ec * 2 * np.pi
        self.beta = beta
        self.N = N
        self.original_dim = original_dim
        self.trunc_dim = trunc_dim
        self.sdim, self.c2dim, self.c1dim = original_dim
        self.total_dim = self.sdim * self.c2dim * self.c1dim
        self.omega_c1 = omega_c1
        self.omega_c2 = omega_c2
        self.g = 0.05 * 2 * np.pi
        self.H, self.H_control = self.get_H()
        self.H_dressed, self.H_control_dressed = self.dressed_basis()

    def sort_eigenpairs(self, eigenvalues, eigenvectors):
        """Sort eigenpairs based on overlap with the basis."""
        n = eigenvectors.shape[0]
        sorted_indices = []
        for i in range(n):
            max_abs_vals = np.abs(eigenvectors[i, :]).copy()  # iterate over rows
            max_index = np.argmax(max_abs_vals)
            while max_index in sorted_indices:
                max_abs_vals[max_index] = -np.inf
                max_index = np.argmax(max_abs_vals)
            sorted_indices.append(max_index)
        sorted_indices = np.array(sorted_indices)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        
        # Fix phase: make the largest component of each eigenvector real and positive
        for j in range(sorted_eigenvectors.shape[1]):
            vec = sorted_eigenvectors[:, j]
            max_idx = np.argmax(np.abs(vec))
            phase = vec[max_idx]
            if np.abs(phase) > 1e-10:
                sorted_eigenvectors[:, j] = vec * np.conj(phase) / np.abs(phase)
        
        return sorted_eigenvalues, sorted_eigenvectors

    def annihilation(self, dim):
        """Annihilation operator."""
        return np.diag(np.sqrt(np.arange(1, dim)), k=1)

    def creation(self, dim):
        """Creation operator."""
        return np.diag(np.sqrt(np.arange(1, dim)), k=-1)

    def tensor_product(self, A, B, C):
        """Tensor product of three matrices."""
        return np.kron(A, np.kron(B, C))

    def cosm(self, matrix):
        return (sci.linalg.expm(1j * matrix) + sci.linalg.expm(-1j * matrix)) / 2


    def SNAIL(self):
        beta = self.beta
        N = self.N
        Ej = self.Ej
        Ec = self.Ec
        phi_ex = self.phi_ex * 2 * np.pi
        sdim = 20

        def U_s(phi):
            return np.asarray(-beta * np.cos(phi - phi_ex) - N * np.cos(phi / N))[0]    
        phi_min = minimize(U_s, np.array([0.]), method='BFGS').x[0]
        self.phi_min = phi_min
        c2 = beta * np.cos(phi_min - phi_ex) + 1 / N * np.cos(phi_min / N)
        omega_s = np.sqrt(8 * c2 * Ej * Ec)

        phi_zpf = np.power(2 * Ec / (Ej * c2), 1 / 4)
        self.phi_zpf = phi_zpf

        g2 = Ej * phi_zpf ** 2 * c2 / 2
        s = self.annihilation(sdim)
        sd = self.creation(sdim)
        x2 = np.matmul(s + sd, s + sd)
        Hs = (omega_s * np.matmul(sd, s)
              - Ej * (beta * self.cosm(phi_zpf * (s + sd) + (phi_min - phi_ex) * np.identity(sdim))
                      + N * self.cosm((phi_zpf * (s + sd) + phi_min * np.identity(sdim)) / N)) - g2 * x2)
        charge_op = -1j * (s - sd) / 2 / phi_zpf
        energy0, U = np.linalg.eigh(Hs)
        energy0, U = self.sort_eigenpairs(energy0, U)
        energy0 = energy0 - energy0[0]
        Ud = U.transpose().conjugate()
        Hs = Ud @ Hs @ U
        charge_op = Ud @ charge_op @ U
        Hs = Hs - Hs[0, 0] * np.identity(sdim)
        Hs = Hs[0:self.sdim, 0:self.sdim]
        charge_op = charge_op[0:self.sdim, 0:self.sdim]
        return Hs, charge_op

    def get_H(self):
        """Constructs the full Hamiltonian."""
        c1dim = 20
        c2dim = 20

        Hs, charge_op = self.SNAIL()
        squid = [Hs, charge_op]

        cavity = scqubits.Oscillator(E_osc=self.omega_c1, truncated_dim=c1dim)
        Hc = np.diag(np.array(cavity.eigenvals(evals_count=c1dim)
                                - cavity.eigenvals(evals_count=c1dim)[0])) * 2 * np.pi
        Hc = Hc[0:self.c1dim, 0:self.c1dim]
        Vc = np.array(cavity.creation_operator() + cavity.annihilation_operator())
        Vc = Vc[0:self.c1dim, 0:self.c1dim]
        cavity = [Hc, Vc]

        cavity2 = scqubits.Oscillator(E_osc=self.omega_c2, truncated_dim=c2dim)
        Hc2 = np.diag(np.array(cavity2.eigenvals(evals_count=c2dim)
                                 - cavity2.eigenvals(evals_count=c2dim)[0])) * 2 * np.pi
        Hc2 = Hc2[0:self.c2dim, 0:self.c2dim]
        Vc2 = np.array(cavity2.creation_operator() + cavity2.annihilation_operator())
        Vc2 = Vc2[0:self.c2dim, 0:self.c2dim]
        cavity2 = [Hc2, Vc2]

        H, H_control = self.composite_sys(squid, cavity, cavity2)
        return H, H_control

    def composite_sys(self, squid, cavity, cavity2):
        """Composes the full system Hamiltonian from subsystems."""
        Hs, charge_op = squid
        Hc, Vc = cavity
        Hc2, Vc2 = cavity2
        sdim = Hs.shape[0]
        cdim = Hc.shape[0]
        cdim2 = Hc2.shape[0]
        Ic = np.identity(cdim)
        Ic2 = np.identity(cdim2)
        Is = np.identity(sdim)
        Hs = self.tensor_product(Hs, Ic, Ic2)
        Hc = self.tensor_product(Is, Hc, Ic2)
        Hc2 = self.tensor_product(Is, Ic, Hc2)
        H_int = self.g * self.tensor_product(charge_op, Vc, Ic2)
        H_int2 = self.g * self.tensor_product(charge_op, Ic, Vc2)
        H = Hs + Hc + Hc2 + H_int + H_int2
        H_control = self.tensor_product(charge_op, Ic, Ic2)
        return H, H_control

    def dressed_basis(self):
        H, H_control = self.get_H()
        evals, U = np.linalg.eigh(H)
        evals, U = self.sort_eigenpairs(evals, U)
        Ud = U.transpose().conjugate()
        H_dressed = Ud @ H @ U
        H_dressed = H_dressed - H_dressed[0, 0] * np.identity(self.total_dim)
        H_control_dressed = Ud @ H_control @ U
        return H_dressed, H_control_dressed

    def omegasder(self):
        """Derivative of energy levels with respect to flux.
        Based on: 2 * Ej * pi * sin(pi * phi_ex) with delta_phi = 1
        """
        return -2 * self.Ej * np.pi * np.sin(np.pi * self.phi_ex)*self.phi_zpf**2

    def der_formula(self, A, omegad):
        Hs, _ = self.SNAIL()
        omegasder = self.omegasder()
        omegas = Hs[1,1]
        anh = -self.Ec
        g = self.g/2/self.phi_zpf
        A = A/2/self.phi_zpf
        Delta = omegas - self.omega_c1*2*np.pi
        omegas += g**2/Delta
        delta = omegas - omegad
        chi = g**2/Delta**2*anh*2
        delta1 = delta + chi
        der = g**2/Delta**2*(1 - 2*A**2*anh*(delta + delta1)/(delta*delta1)**2/4)
        # der = g**2/Delta**2*(1 - A**2/delta**3*anh)
        return der*omegasder
    
    def der_formula2(self, A, omegad):
        Hs, _ = self.SNAIL()
        omegasder = self.omegasder()
        position = self.state_index((1, 0, 0), self.original_dim)
        omegas = self.H_dressed[position, position]
        anh = -self.Ec
        g = self.g/2/self.phi_zpf
        A = A/2/self.phi_zpf
        Delta = omegas - self.omega_c1*2*np.pi
        delta = omegas - omegad
        chi = g**2/Delta**2*anh*2
        delta1 = delta + chi
        der = g**2/Delta**2*(1 - 2*A**2*anh*(delta + delta1)/(delta*delta1)**2/4)
        # der = g**2/Delta**2*(1 - A**2/delta**3*anh)
        return der*omegasder


    def quasi_energy(self, A=0e-3 * 2 * np.pi, omega=6.185 * 2 * np.pi):
        """Calculates the quasi-energies using Qutip."""
        H0, Hc = self.H_dressed, self.H_control_dressed
        # Convert to numpy for Qutip
        H0_np = np.array(H0)
        Hc_np = np.array(Hc)

        # Create Qutip objects
        H0_qt = qt.Qobj(H0_np)
        Hc_qt = qt.Qobj(Hc_np)

        T = (2 * np.pi) / omega
        H = [H0_qt, [Hc_qt, lambda t, args: A * np.cos(omega * t)]]

        # Calculate Floquet modes using FloquetBasis
        floquet_basis = qt.FloquetBasis(H, T)
        f_modes = floquet_basis.mode(0)  # modes at t=0
        f_energies = floquet_basis.e_quasi

        # Convert modes to eigenvector matrix
        evecs_np = np.column_stack([m.full() for m in f_modes])
        evals_np = np.array(f_energies)

        sorted_evals, sorted_evecs = self.sort_eigenpairs(evals_np, evecs_np)

        e_quasi = sorted_evals
        e_quasi = e_quasi - e_quasi[0]
        if self.trunc_dim[1] == 1:
            return e_quasi[1], e_quasi[2], e_quasi[3]
        return e_quasi[1], e_quasi[self.trunc_dim[2]]
    
    def quasi_energy2(self, A=0e-3 * 2 * np.pi, omega=6.185 * 2 * np.pi):
        """Calculates the quasi-energies using Qutip."""
        H0, Hc = self.H_dressed, self.H_control_dressed
        # Convert to numpy for Qutip
        H0_np = np.array(H0)
        Hc_np = np.array(Hc)

        # Create Qutip objects
        H0_qt = qt.Qobj(H0_np)
        Hc_qt = qt.Qobj(Hc_np)

        T = (2 * np.pi) / omega
        H = [H0_qt, [Hc_qt, lambda t, args: A * np.cos(omega * t)]]

        # Calculate Floquet modes using FloquetBasis
        floquet_basis = qt.FloquetBasis(H, T)
        f_modes = floquet_basis.mode(0)  # modes at t=0
        f_energies = floquet_basis.e_quasi

        # Convert modes to eigenvector matrix
        evecs_np = np.column_stack([m.full() for m in f_modes])
        evals_np = np.array(f_energies)

        sorted_evals, sorted_evecs = self.sort_eigenpairs(evals_np, evecs_np)

        e_quasi = sorted_evals
        e_quasi = e_quasi - e_quasi[0]
        if self.trunc_dim[1] == 1:
            return e_quasi[1], e_quasi[2], e_quasi[3]
        return e_quasi[1], e_quasi[self.trunc_dim[2]]


    def equasi_gradient(self, A, omega):
        """Gradient of the quasi-energy with respect to flux using finite differences."""
        delta = 1e-6
        phi = self.phi_ex

        def get_energies(p):
            sc = Hamiltonian(p, self.Ej / (2 * np.pi), self.Ec / (2 * np.pi),
                             self.original_dim, self.trunc_dim, self.omega_c1,
                             self.omega_c2)
            return np.array(sc.quasi_energy(A, omega))

        E_plus = get_energies(phi + delta)
        E_minus = get_energies(phi - delta)

        grad = (E_plus - E_minus) / (2 * delta)
        return grad

    def static_rate(self, der):
        """Calculates the static rate."""
        A = 5e-6
        # Handle the case where der is a tuple of derivatives
        if isinstance(der, (tuple, list, np.ndarray)):
            results = []
            for d in der:
                results.append(np.abs(d * 4.4) * 1e6 * A)
            return results
        else:
            return np.abs(der * 4.4) * 1e6 * A  # convert from GHz to MHz, multiplied by A

    def state_index(self, index, dim):
        """Convert a 3D index tuple (n, k, m) to a 1D matrix position.
        
        Args:
            index: Tuple (n, k, m) representing the state indices for (squid, cavity2, cavity1)
            dim: Tuple (N, K, M) representing the dimensions
            
        Returns:
            1D index position in the flattened matrix
        """
        n, k, m = index
        N, K, M = dim
        return n * K * M + k * M + m

    def fourth_order_der(self, A, omegad):
        """Fourth-order energy derivative w.r.t. flux via chain rule.
        
        Computes d(E)/d(phi) = d(E)/d(Delta) * d(omegas)/d(phi),
        where E is the fourth-order perturbative energy correction and
        Delta = omegad - omegas.
        """
        Hs, _ = self.SNAIL()
        omegasder = self.omegasder()
        position = self.state_index((1, 0, 0), self.original_dim)
        omegas = self.H_dressed[position, position]
        A = A/2
        # System parameters
        g = self.g / 2 / self.phi_zpf
        A = A / 2 / self.phi_zpf
        K_a = -self.Ec  # anharmonicity (Kerr)
        Delta_cav = omegas - self.omega_c1 * 2 * np.pi  # SNAIL-cavity detuning
        chi = g**2 / Delta_cav**2 * K_a * 2  # dispersive shift

        # Delta = omegad - omegas (drive - SNAIL detuning)
        Delta =   omegas - omegad

        def energy_expr(D):
            """Fourth-order perturbative energy expression."""
            t1 = 2 * A**4 / ((-D - chi)**2 * (-2 * D - K_a - 2 * chi))
            t2 = -A**4 / (-D - chi)**3
            t3 = -2 * A**4 / (D**2 * (-2 * D - K_a))
            t4 = -A**4 / D**3
            t5 = A**2 / (-D - chi)
            t6 = A**2 / D
            return t1+t2+t3+t4+t5+t6

        # Finite difference derivative w.r.t. Delta
        dD = 1e-6
        dE_dDelta = (energy_expr(Delta + dD) - energy_expr(Delta - dD)) / (2 * dD) + g**2/Delta_cav**2

        # Chain rule: d(E)/d(phi) = d(E)/d(Delta) * d(Delta)/d(phi)
        # Delta = omegad - omegas  =>  d(Delta)/d(phi) = -d(omegas)/d(phi)
        return dE_dDelta * omegasder

    def omegas_app(self):
        """Approximate dressed SNAIL frequency."""
        Hs, _ = self.SNAIL()
        omegas = Hs[1, 1]
        g = self.g / 2 / self.phi_zpf
        Delta = omegas - self.omega_c1 * 2 * np.pi
        omegas += g ** 2 / Delta
        return omegas

    def g3(self):
        """Third-order nonlinear coupling of the SNAIL.

        c3 = (N^2 - 1) / N^2 * sin(phi_min / N)
        g3 = Ej * phi_zpf^3 * c3 / 3!

        Returns g3 as a float in angular frequency units (rad * GHz).
        """
        N = self.N
        phi_min = float(np.squeeze(self.phi_min))
        phi_zpf = float(np.squeeze(self.phi_zpf))
        c3 = (N**2 - 1) / N**2 * np.sin(phi_min / N)
        g3_val = self.Ej * phi_zpf**3 * c3 / 6
        return float(np.real(g3_val))

    def chi(self):
        """Dispersive shift between the SNAIL and cavity 1.

        chi = E(1,0,1) - E(1,0,0) - E(0,0,1)

        where H_dressed is already shifted so E(0,0,0) = 0.
        Returns chi in angular frequency units (rad * GHz).
        """
        Hd = self.H_dressed
        dim = self.original_dim
        E001 = float(np.real(Hd[self.state_index((0, 0, 1), dim), self.state_index((0, 0, 1), dim)]))
        E100 = float(np.real(Hd[self.state_index((1, 0, 0), dim), self.state_index((1, 0, 0), dim)]))
        E101 = float(np.real(Hd[self.state_index((1, 0, 1), dim), self.state_index((1, 0, 1), dim)]))
        return E101 - E100 - E001

    def anharmonicity(self):
        """SNAIL anharmonicity (self-Kerr).

        K = E(2,0,0) - 2 * E(1,0,0)

        Returns K in angular frequency units (rad * GHz).
        """
        Hd = self.H_dressed
        dim = self.original_dim
        E100 = float(np.real(Hd[self.state_index((1, 0, 0), dim), self.state_index((1, 0, 0), dim)]))
        E200 = float(np.real(Hd[self.state_index((2, 0, 0), dim), self.state_index((2, 0, 0), dim)]))
        return E200 - 2 * E100
