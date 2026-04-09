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
            shape attributes, e.g. scipy.sparse.csr_matrix). The matrix must
            be square. For symmetric types (SPD, symmetric indefinite), pass
            only the upper triangle in CSR format. For structurally symmetric
            and nonsymmetric types, pass the full matrix.
        mtype: Matrix type constant (MTYPE_REAL_SYM_POSDEF, etc.).
        iparms: Optional dict of {index: value} iparm overrides.
        msglvl: Message level (0 = silent, 1 = print statistics).
    """

    def __init__(self, A, mtype, iparms=None, msglvl=0):
        if len(A.shape) != 2:
            raise ValueError("A must be a 2D sparse matrix")
        n, n_cols = A.shape
        if n != n_cols:
            raise ValueError("A must be square")

        indptr = np.asarray(A.indptr, dtype=np.int64)
        indices = np.asarray(A.indices, dtype=np.int64)
        data = np.asarray(A.data, dtype=np.float64)

        if mtype in _SYMMETRIC_MTYPES:
            for row in range(n):
                row_start = indptr[row]
                row_end = indptr[row + 1]
                if np.any(indices[row_start:row_end] < row):
                    raise ValueError(
                        "symmetric matrix types require upper-triangular CSR data only"
                    )

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

    def refactor(self, values):
        """Re-factorize with new values (same sparsity pattern).

        Runs only numeric factorization (phase 22). Does not re-run symbolic
        analysis. Raises if symbolic analysis is invalid; use factor() to
        re-analyze from scratch.

        Values must match the stored sparsity pattern exactly.
        """
        self._solver.refactor_values(np.asarray(values, dtype=np.float64))

    def factor(self, values):
        """Re-analyze and re-factorize with new values (phases 11 + 22).

        This always performs fresh symbolic analysis before numeric
        factorization. Use this for error recovery or when iparm changes
        require fresh symbolic analysis. For normal refactoring, use
        refactor().

        Values must match the stored sparsity pattern exactly.
        """
        self._solver.factor(np.asarray(values, dtype=np.float64))

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
