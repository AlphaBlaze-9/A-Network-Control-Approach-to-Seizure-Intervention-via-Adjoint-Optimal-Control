"""
connectome.py  --  Phase 1 / Week 1: Structural Network Acquisition & Scale
===========================================================================
Loads the HCP-MMP (Glasser 360) structural connectome, cleans it, and applies
the spectral normalisation used in the original ``week1.py`` (divide by the
largest-magnitude eigenvalue so that the leading eigenvalue of A is 1). Also
provides the reduced, seizure-relevant sub-network used for fast development of
the heavy optimal-control weeks.

The connectome here is what makes every downstream result "connectome-grounded":
the same matrix A feeds the coupling term of the Hopf network, the propagation
study, and the optimal controller.
"""

from __future__ import annotations
import numpy as np
import scipy.io

from . import config


def load_raw_connectome(mat_path: str = config.MAT_PATH):
    """Load the raw HCP-MMP connectivity matrix and its region labels.

    Returns
    -------
    C : (360, 360) float ndarray   -- raw streamline-count connectivity
    labels : list[str]             -- Glasser region names (e.g. 'L_V1')
    """
    mat = scipy.io.loadmat(mat_path)
    C = np.asarray(mat["connectivity"], dtype=float)
    # Region labels are stored as a single NUL-terminated, newline-joined byte
    # string in the 'name' field.
    raw = bytes(mat["name"].ravel().tolist()).split(b"\x00")[0].decode(errors="replace")
    labels = [s for s in raw.split("\n") if s]      # drop trailing empty token
    return C, labels


def clean_and_normalise(C: np.ndarray) -> np.ndarray:
    """Clean and spectrally normalise the adjacency matrix.

    Steps (mirroring the original week1.py, made robust):
      1. Zero the diagonal (no self-connections).
      2. Symmetrise (structural connectomes are undirected; guards float noise).
      3. Divide by the largest |eigenvalue| so spectral radius of A is 1.0.

    Spectral normalisation puts every connectome on the same dynamical footing:
    the global coupling factor G in the Hopf model then has a comparable meaning
    across subjects/atlases.
    """
    A = C.astype(float).copy()
    np.fill_diagonal(A, 0.0)
    A = 0.5 * (A + A.T)                              # enforce exact symmetry
    eig = np.linalg.eigvalsh(A)                      # symmetric -> real spectrum
    radius = np.max(np.abs(eig))
    if radius > 0:
        A = A / radius
    return A


def degree_vector(A: np.ndarray) -> np.ndarray:
    """Weighted degree d_i = sum_j A_ij (used by the diffusive coupling term)."""
    return A.sum(axis=1)


def select_reduced_subnetwork(A: np.ndarray, labels, size: int = config.REDUCED_SIZE,
                              focus: int = config.FOCUS_NODE):
    """Pick a dense sub-network centred on the seizure focus for fast dev runs.

    Strategy: keep the focus node plus the ``size-1`` regions most strongly
    connected to it (largest A[focus, :] weights). This preserves the part of
    the network that actually carries seizure spread, which is exactly what the
    reduced scale needs to remain a faithful (if smaller) testbed.

    Returns
    -------
    A_sub : (size, size) ndarray  -- re-normalised reduced adjacency
    idx   : (size,) int ndarray   -- original node indices kept (sorted)
    sub_labels : list[str]
    sub_focus  : int              -- index of the focus node *within* A_sub
    """
    strength = A[focus].copy()
    strength[focus] = np.inf                         # force-include the focus
    idx = np.sort(np.argsort(strength)[::-1][:size])
    A_sub = A[np.ix_(idx, idx)].copy()
    # Re-normalise the sub-network so its spectral radius is again 1.0.
    r = np.max(np.abs(np.linalg.eigvalsh(A_sub)))
    if r > 0:
        A_sub /= r
    sub_labels = [labels[i] for i in idx]
    sub_focus = int(np.where(idx == focus)[0][0])
    return A_sub, idx, sub_labels, sub_focus


def get_network(scale: str | None = None):
    """Return (A, labels, focus_index) for the active scale.

    This is the single entry point the rest of the project uses, so switching
    between FULL and REDUCED is just an environment variable (SCALE=reduced).
    """
    scale = scale or config.ACTIVE_SCALE
    C, labels = load_raw_connectome()
    A = clean_and_normalise(C)
    if scale == "reduced":
        A_sub, idx, sub_labels, sub_focus = select_reduced_subnetwork(A, labels)
        return A_sub, sub_labels, sub_focus
    return A, labels, config.FOCUS_NODE


if __name__ == "__main__":
    C, labels = load_raw_connectome()
    A = clean_and_normalise(C)
    np.save(config.A_MATRIX_NPY, A.astype(np.float32))
    print(f"Loaded {C.shape[0]}-node {config.ATLAS_NAME} connectome.")
    print(f"  density            : {np.count_nonzero(A)/A.size:6.3f}")
    print(f"  spectral radius (A): {np.max(np.abs(np.linalg.eigvalsh(A))):6.3f}")
    print(f"  focus region       : idx {config.FOCUS_NODE} = {labels[config.FOCUS_NODE]}")
    print(f"  saved normalised A -> {config.A_MATRIX_NPY}")
