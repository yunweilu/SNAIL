import jax.numpy as jnp

def commutator(A, B):
    return A @ B - B @ A

def nested_commutator(A, B, n):
    if n == 1:
        return commutator(A, B)
    else:
        return commutator(A, nested_commutator(A, B, n-1))

def each_level_its_own_subspace(dim):
    """
    Obtain the indices of the subspace to consider.

    Args:
        dim (int): The dimension of the subspace.

    Returns:
        list: A list of lists, each containing a single index of the subspace to consider.
    """
    # Define the indices of the subspace to consider
    subspace_indices = [[i] for i in range(dim)]
    return subspace_indices

def create_subspace_projectors(dim, subspace_indices):
    """
    Create projectors for the given subspaces using jax.numpy.

    Args:
        dim (int): The total dimension of the Hilbert space.
        subspace_indices (list): A list of lists, each containing the indices of a subspace.

    Returns:
        list: A list of projectors (jax.numpy arrays) for the given subspaces.
    """
    projectors = []
    
    for subspace in subspace_indices:
        # Initialize the projector matrix P as a zero matrix of size dim x dim
        P = jnp.zeros((dim, dim))
        
        # Create the basis states and compute the projector for each index in the subspace
        for idx in subspace:
            # Create a one-hot basis state vector with 1 at the index 'idx'
            basis_state = jnp.eye(dim)[idx]  # shape (dim,)
            # Compute the rank-1 projector: outer product of the basis state with itself
            P = P + jnp.outer(basis_state, basis_state)
        
        # Add the projector to the list
        projectors.append(P)
    
    return projectors

def swt_subspace(H0, V, subspace_indices = None, order = 2):
    """
    Compute the Schrieffer-Wolff transformation on a subspace of the Hamiltonian.

    Args:
        H0 (jnp.array): The unperturbed Hamiltonian.
        V (jnp.array): The perturbation Hamiltonian.
        subspace_indices (list): The indices of the subspace to consider.

    Returns:
        tuple: The Schrieffer-Wolff transformation operators ([S1]) and 
               the transformed Hamiltonian components ([H1, H2]).
    """
    dim = H0.shape[0]
    if subspace_indices == None:
        subspace_indices = each_level_its_own_subspace(dim)
    projectors = create_subspace_projectors(dim, subspace_indices)
    Vd = jnp.zeros((dim, dim))
    for P in projectors:
        Vd = Vd + (P @ V @ P)
    Vod = V - Vd
    # Compute the energy differences in the subspace
    delta = jnp.array([[H0[i, i] - H0[j, j] if i != j else 1 for j in range(dim)] for i in range(dim)])
    
    # Compute the Schrieffer-Wolff transformation components
    H1 = Vd
    S1 = jnp.array([[Vod[i, j] / delta[i, j] for j in range(dim)] for i in range(dim)])
    H2 = 0.5 * commutator(S1, Vod)
    
    S2 = jnp.array([[-commutator(Vd, S1)[i, j] / delta[i, j] for j in range(dim)] for i in range(dim)])
    H3 = 0.5 * commutator(S2, Vod)
    S3 = jnp.array([[(commutator(S2, Vd)[i, j] + (1/3) * nested_commutator(S1, Vod, 2)[i, j]) / delta[i, j] for j in range(dim)] for i in range(dim)])
    H4 = 0.5 * commutator(S3, Vod) - (1/24) * nested_commutator(S1, Vod, 3)
    if order == 2:
        evals_app = jnp.diag(H0 + H1 + H2)
    if order == 3:
        evals_app = jnp.diag(H0 + H1 + H2 + H3)
    if order == 4:
        evals_app = jnp.diag(H0 + H1 + H2 + H3 + H4)
    evals_app = evals_app - evals_app[0]
    return  jnp.real(evals_app)