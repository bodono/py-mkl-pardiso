"""Python wrapper for Intel oneMKL PARDISO sparse direct solver."""

import numpy as np

from pymklpardiso._mkl_pardiso import (
    MTYPE_REAL_NONSYM,
    MTYPE_REAL_STRUCT_SYM,
    MTYPE_REAL_SYM_INDEF,
    MTYPE_REAL_SYM_POSDEF,
)
from pymklpardiso._mkl_pardiso import PardisoSolver as _PardisoSolver

_SYMMETRIC_MTYPES = (MTYPE_REAL_SYM_POSDEF, MTYPE_REAL_SYM_INDEF)


class PardisoSolver:
    """PARDISO sparse direct solver.

    Args:
        A: Sparse matrix in CSR format (any object with indptr, indices, data,
            shape attributes, e.g. scipy.sparse.csr_matrix). For symmetric
            types (SPD, symmetric indefinite), the upper triangle is extracted
            automatically — you may pass either the full matrix or just the
            upper triangle. For structurally symmetric and nonsymmetric types,
            pass the full matrix.
        mtype: Matrix type constant (MTYPE_REAL_SYM_POSDEF, etc.).
        iparms: Optional dict of {index: value} iparm overrides.
        msglvl: Message level (0 = silent, 1 = print statistics).
    """

    def __init__(self, A, mtype, iparms=None, msglvl=0):
        indptr = np.asarray(A.indptr, dtype=np.int64)
        indices = np.asarray(A.indices, dtype=np.int64)
        data = np.asarray(A.data, dtype=np.float64)
        n = A.shape[0]


        if mtype in _SYMMETRIC_MTYPES:
            # PARDISO expects only the upper triangle for symmetric types.
            rows = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))

            # Using a boolean mask is typically faster/cleaner for indexing than np.where
            mask = indices >= rows

            # Save state (using np.where here if your class needs integer indices later)
            self._triu_mask = np.where(mask)[0]
            self._full_nnz = len(data)

            # Reconstruct indptr directly into a new array
            new_indptr = np.zeros(n + 1, dtype=np.int64)
            new_indptr[1:] = np.cumsum(np.bincount(rows[mask], minlength=n))

            # Reassign to the expected variables
            indptr, indices, data = new_indptr, indices[mask], data[mask]
        else:
            self._triu_mask = None
            self._full_nnz = None

        self._solver = _PardisoSolver(mtype, msglvl)
        if iparms:
            for idx, val in iparms.items():
                self._solver.set_iparm(idx, val)
        self._solver.set_pattern(ia=indptr, ja=indices, n=n)
        self._solver.factor(data)

    def solve(self, b):
        """Solve Ax = b. Accepts 1D (n,) or 2D (n, nrhs) arrays."""
        return self._solver.solve(b)

    def solve_into(self, b, x):
        """Solve Ax = b writing into pre-allocated x."""
        self._solver.solve_into(b, x)

    def _extract_triu_values(self, values):
        """Extract upper-triangle values if symmetric and full data is passed."""
        values = np.asarray(values, dtype=np.float64)
        if self._triu_mask is not None and len(values) == self._full_nnz:
            values = values[self._triu_mask]
        return values

    def refactor(self, values):
        """Re-factorize with new values (same sparsity pattern).

        Runs only numeric factorization (phase 22). Does not re-run symbolic
        analysis. Use factor() to re-analyze from scratch.

        For symmetric types, accepts either full-matrix or upper-triangle data.
        """
        self._solver.refactor_values(self._extract_triu_values(values))

    def factor(self, values):
        """Re-analyze and re-factorize with new values (phases 11 + 22).

        Use this for error recovery or when iparm changes require fresh
        symbolic analysis. For normal refactoring, use refactor().

        For symmetric types, accepts either full-matrix or upper-triangle data.
        """
        self._solver.factor(self._extract_triu_values(values))

    # -- iparm access --

    def set_iparm(self, idx, value):
        """Set a single iparm entry."""
        self._solver.set_iparm(idx, value)

    def get_iparm(self):
        """Get all 64 iparm values."""
        return self._solver.get_iparm()

    def get_iparm_value(self, idx):
        """Get a single iparm value."""
        return self._solver.get_iparm_value(idx)

    def set_iparm_all(self, iparm):
        """Set all 64 iparm values."""
        self._solver.set_iparm_all(iparm)

    # -- permutation --

    def set_perm(self, perm):
        """Set fill-reducing permutation."""
        self._solver.set_perm(perm)

    def clear_perm(self):
        """Clear permutation."""
        self._solver.clear_perm()

    def has_perm(self):
        """Whether a permutation is set."""
        return self._solver.has_perm()

    # -- low-level phase control --

    def run_phase(self, phase):
        """Run an arbitrary PARDISO phase."""
        self._solver.run_phase(phase)

    def run_phase_into(self, phase, b, x):
        """Run a PARDISO phase with RHS/output arrays."""
        self._solver.run_phase_into(phase, b, x)

    # -- lifecycle --

    def release(self):
        """Free PARDISO internal memory."""
        self._solver.release()

    def set_msglvl(self, msglvl):
        """Change message level."""
        self._solver.set_msglvl(msglvl)

    # -- properties --

    @property
    def n(self):
        """Matrix dimension."""
        return self._solver.n()

    @property
    def nnz(self):
        """Number of nonzeros in the sparsity pattern."""
        return self._solver.nnz()

    @property
    def mtype(self):
        """Matrix type."""
        return self._solver.mtype()


__all__ = [
    "PardisoSolver",
    "MTYPE_REAL_STRUCT_SYM",
    "MTYPE_REAL_SYM_INDEF",
    "MTYPE_REAL_SYM_POSDEF",
    "MTYPE_REAL_NONSYM",
]
