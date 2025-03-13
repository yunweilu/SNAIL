from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import scipy as sci
import matplotlib.pyplot as plt
import qutip as qt

def annihilation(dim):
    return np.diag(np.sqrt(np.arange(1,dim)),1)
def creation(dim):
    return np.diag(np.sqrt(np.arange(1,dim)),-1)
def sort_eigenpairs(eigenvalues, eigenvectors):
    n = eigenvectors.shape[0]
    sorted_indices = []

    for i in range(n):
        max_abs_vals = np.abs(eigenvectors[i, :])
        max_index = np.argmax(max_abs_vals)
        while max_index in sorted_indices:
            max_abs_vals[max_index] = -np.inf
            max_index = np.argmax(max_abs_vals)
        sorted_indices.append(max_index)

    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    return sorted_eigenvalues, sorted_eigenvectors
def SNAIL(phi_ex,beta,N,Ej,Ec):
    phi_ex = phi_ex*2*np.pi
    def Us_min(phi_ex):
        def U_s(phi): 
            return (-beta*np.cos(phi-phi_ex)-N*np.cos((phi)/N))
        phi_min = sci.optimize.minimize(U_s,0).x
        return phi_min
    
    def phi_minde(ans, phi_ex):
        def phi_minde_vjp(g):
            c2 = beta*np.cos(ans - phi_ex) + 1/N*np.cos(ans/N)
            return g*beta*np.cos(ans - phi_ex)/c2
        return phi_minde_vjp
    
    phi_min = Us_min(phi_ex)
    # potential expansion around minimum
    c2 = beta*np.cos(phi_min - phi_ex) + 1/N*np.cos(phi_min/N)
    c3 = -(N**2-1)/N**2*beta*np.sin(phi_min-phi_ex)
    
    omega_s = np.sqrt(8*c2*Ej*Ec)
    phi_zpf = np.power(2*Ec/(Ej*c2),1/4)
    g2 = Ej*phi_zpf**2*c2/2
    g3 = Ej * phi_zpf ** 3 * c3 / 3 / 2
    sdim = 10
    s = annihilation(sdim)
    sd = creation(sdim)
    x2 = np.matmul(s+sd,s+sd)
    Hs =( omega_s * np.matmul(sd,s)
        - Ej*(beta*sci.linalg.cosm(phi_zpf*(s+sd)+(phi_min-phi_ex)*np.identity(sdim))
        + N*sci.linalg.cosm((phi_zpf*(s+sd)+phi_min*np.identity(sdim))/N))- g2*x2)
    charge_op = -1j*(s-sd)/2/phi_zpf
    energy0,U = np.linalg.eigh(Hs)
    energy0,U = sort_eigenpairs(energy0, U)
    energy0 = energy0 - energy0[0]
    Ud = U.transpose().conjugate()
    Hs = Ud@Hs@U
    Hs = Hs - Hs[0,0]*np.identity(sdim)
    return Hs,Ud@charge_op@U, phi_zpf

def composite_sys(squid,cavity,  ):
    Hs, charge_op = squid
    Hc, Vc = cavity
    sdim = Hs.shape[0]
    cdim = Hc.shape[0]
    
    Ic = np.identity(cdim)
    Is = np.identity(sdim)
    Hs = np.kron(Hs, Ic)
    Hc = np.kron(Is, Hc)
    H_int = 0.05*2*np.pi * np.kron(charge_op, Vc) 
    H = Hs + Hc + H_int
    H_control = np.kron(charge_op, Ic)
    return H, H_control


def state_index(index,dim):
    n,k = index
    N,K = dim
    return n*K+k

import qutip as qt
def find_optimal_k(A, B, D):
    # Define a large initial minimum difference
    min_diff = float('inf')
    optimal_k = None
    
    # Iterate over a range of possible k values
    # The range can be adjusted based on expected size of k or other insights you have about your problem
    for k in range(-1000, 1000):
        # Calculate the difference for this value of k
        diff = abs(A - (B + k * D))
        
        # If this is the smallest difference we've found so far, update min_diff and optimal_k
        if diff < min_diff:
            min_diff = diff
            optimal_k = k
            
    return optimal_k
# Function to calculate overlap (you might use inner product, fidelity, etc.)
def calculate_overlap(state1, state2):
    return abs((state1.dag() * state2))**2


def calculate_floquet_energies(A, omega,H0, Hc):
    # Define system parameters
    dim = [10, 6]
    index01 = state_index([0,1], dim)
    index10 = state_index([1,0], dim)
    index11 = state_index([1,1], dim)
    index02 = state_index([0,2], dim)
    index03 = state_index([0,3], dim)
    omega01 = np.diag(H0)[index01]
    omega10 = np.diag(H0)[index10]
    omega11 = np.diag(H0)[index11]
    omega02 = np.diag(H0)[index02]
    omega03 = np.diag(H0)[index03]
    H0 = qt.Qobj(H0)
    Hc = qt.Qobj(Hc)

    T = (2 * np.pi) / omega

    # Define the Hamiltonian
    H = [H0, [Hc, lambda t, args: A * np.cos(args['w'] * t)]]

    # Set up the Floquet solver
    floquet_basis = qt.FloquetBasis(H, T, args={'w': omega})

    # Compute Floquet modes and energies
    f_modes = floquet_basis.mode(0)
    f_energies = floquet_basis.e_quasi

    # Define basis states
    basis_states = [qt.basis(H0.dims[0][0], 0), 
                    qt.basis(H0.dims[0][0], index01),
                    qt.basis(H0.dims[0][0], index10),
                    qt.basis(H0.dims[0][0], index11),
                    qt.basis(H0.dims[0][0], index02),
                    qt.basis(H0.dims[0][0], index03)]

    # Find Floquet states with maximum overlap
    max_overlap_indices = [-1] * 6
    max_overlaps = [0] * 6
    for f_index, f_state in enumerate(f_modes):
        for b_index, b_state in enumerate(basis_states):
            overlap = calculate_overlap(f_state, b_state)
            if overlap > max_overlaps[b_index]:
                max_overlaps[b_index] = overlap
                max_overlap_indices[b_index] = f_index

    # Calculate energies
    energy00 = f_energies[max_overlap_indices[0]] / (2 * np.pi)

    energy01 = f_energies[max_overlap_indices[1]] / (2 * np.pi)
    k = find_optimal_k(omega01 / (2 * np.pi), energy01, omega / (2 * np.pi))
    energy01 = energy01 + k * omega / (2 * np.pi) - energy00

    energy10 = f_energies[max_overlap_indices[2]] / (2 * np.pi)
    k = find_optimal_k(omega10 / (2 * np.pi), energy10, omega / (2 * np.pi))
    energy10 = energy10 + k * omega / (2 * np.pi) - energy00

    energy11 = f_energies[max_overlap_indices[3]] / (2 * np.pi)
    k = find_optimal_k(omega11 / (2 * np.pi), energy11, omega / (2 * np.pi))
    energy11 = energy11 + k * omega / (2 * np.pi) - energy00
    # chi = energy11 - energy10 - energy01 

    energy02 = f_energies[max_overlap_indices[4]] / (2 * np.pi)
    k = find_optimal_k(omega02 / (2 * np.pi), energy02, omega / (2 * np.pi))
    energy02 = energy02 + k * omega / (2 * np.pi) - energy00
    # self_kerr = energy02 - 2 * energy01
    
    energy03 = f_energies[max_overlap_indices[5]] / (2 * np.pi)
    k = find_optimal_k(omega03 / (2 * np.pi), energy03, omega / (2 * np.pi))
    energy03 = energy03 + k * omega / (2 * np.pi) - energy00
    return energy01*2*np.pi , energy02*2*np.pi, energy03*2*np.pi,
