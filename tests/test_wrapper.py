"""Tests for the high-level PardisoSolver wrapper in pymklpardiso."""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

from pymklpardiso import (
    MTYPE_REAL_NONSYM,
    MTYPE_REAL_STRUCT_SYM,
    MTYPE_REAL_SYM_INDEF,
    MTYPE_REAL_SYM_POSDEF,
    PardisoSolver,
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spd_upper_csr(A_full):
    """Return upper-triangular CSR representation of a symmetric matrix."""
    M = sp.csr_matrix(np.triu(A_full))
    M.sort_indices()
    return M


def _make_spd(n, density=0.05, seed=42):
    """Generate a random n x n SPD matrix and its upper-triangular CSR."""
    rng = np.random.default_rng(seed)
    R = sp.random(n, n, density=density, random_state=rng, format="csr")
    M = (R.T @ R + sp.eye(n)).tocsr()
    M_upper = sp.triu(M, format="csr")
    M_upper.sort_indices()
    return M.toarray(), M_upper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def A4():
    """4x4 SPD matrix and its upper-triangle CSR."""
    A_full = np.array([
        [10.0, 1.0, 0.0, 0.0],
        [1.0, 8.0, 2.0, 0.0],
        [0.0, 2.0, 12.0, 1.0],
        [0.0, 0.0, 1.0, 6.0],
    ])
    A_upper = _spd_upper_csr(A_full)
    return A_full, A_upper


@pytest.fixture
def solver4(A4):
    """PardisoSolver (wrapper) factored for the 4x4 SPD fixture."""
    A_full, A_upper = A4
    return PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_basic_construction(self, A4):
        A_full, A_upper = A4
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)
        assert solver.n == 4
        assert solver.nnz == A_upper.nnz
        assert solver.mtype == MTYPE_REAL_SYM_POSDEF

    def test_construction_with_iparms(self, A4):
        _, A_upper = A4
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF, iparms={7: 2})
        assert solver.get_iparm_value(7) == 2

    def test_construction_with_msglvl(self, A4):
        _, A_upper = A4
        # msglvl=1 should not error
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF, msglvl=1)
        assert solver.n == 4

    def test_constructor_factors_immediately(self, solver4, A4):
        """Constructor should analyze and factor; solver is ready to solve."""
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_constructor_rejects_rectangular_matrix(self):
        A = sp.csr_matrix(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with pytest.raises(ValueError, match="square"):
            PardisoSolver(A, MTYPE_REAL_NONSYM)

    def test_constructor_rejects_lower_triangle_for_symmetric_type(self):
        A = sp.csr_matrix(np.array([[4.0, 0.0], [1.0, 3.0]]))
        A.sort_indices()
        with pytest.raises(ValueError, match="upper-triangular"):
            PardisoSolver(A, MTYPE_REAL_SYM_POSDEF)


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

class TestSolve:
    def test_solve_1d(self, solver4, A4):
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_solve_2d(self, solver4, A4):
        A_full, _ = A4
        B = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        X = solver4.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)

    def test_solve_into_1d(self, solver4, A4):
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(4)
        solver4.solve_into(b, x)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_solve_into_2d(self, solver4, A4):
        A_full, _ = A4
        B = np.asfortranarray(
            np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        )
        X = np.asfortranarray(np.zeros((4, 2)))
        solver4.solve_into(B, X)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)

    def test_solve_into_rejects_non_float64_output(self, solver4):
        b = np.ones(4)
        x = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="float64"):
            solver4.solve_into(b, x)

    def test_solve_into_rejects_readonly_output(self, solver4):
        b = np.ones(4)
        x = np.zeros(4)
        x.setflags(write=False)
        with pytest.raises(ValueError, match="writable"):
            solver4.solve_into(b, x)


# ---------------------------------------------------------------------------
# Refactor (phase 22 only)
# ---------------------------------------------------------------------------

class TestRefactor:
    def test_refactor_correctness(self, solver4, A4):
        A_full, _ = A4
        A2_full = A_full.copy()
        A2_full[0, 0] += 2.0
        A2_full[1, 1] += 1.0
        A2_full[2, 2] += 3.0
        A2_full[3, 3] += 0.5
        A2_upper = _spd_upper_csr(A2_full)

        solver4.refactor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)

    def test_repeated_refactor(self, A4):
        A_full, A_upper = A4
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(10):
            A_mod = A_full.copy()
            A_mod[0, 0] += 0.5 * i
            A_mod[2, 2] += 0.3 * i
            A_csr = _spd_upper_csr(A_mod)
            solver.refactor(A_csr.data.astype(np.float64))
            x = solver.solve(b)
            npt.assert_allclose(A_mod @ x, b, atol=1e-12)

    def test_refactor_only_phase_22(self, A4):
        """Refactor should keep working when only numeric values change."""
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = _spd_upper_csr(np.array([
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 2.0],
            [0.0, 2.0, 5.0],
        ]))

        solver = PardisoSolver(
            A_upper, MTYPE_REAL_SYM_INDEF,
            iparms={9: 8, 10: 1, 12: 1},
        )

        # Refactor with modified values
        A2_full = np.array([
            [5.0, 1.0, 0.0],
            [1.0, -2.0, 2.0],
            [0.0, 2.0, 6.0],
        ])
        A2_upper = sp.csr_matrix(np.triu(A2_full))
        A2_upper.sort_indices()
        solver.refactor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)

    def test_refactor_requires_reanalysis_after_release(self, A4):
        _, A_upper = A4
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)
        solver.release()
        with pytest.raises(RuntimeError, match="call factor"):
            solver.refactor(A_upper.data.astype(np.float64))


# ---------------------------------------------------------------------------
# Factor (phases 11 + 22)
# ---------------------------------------------------------------------------

class TestFactor:
    def test_factor_re_analyzes(self, solver4, A4):
        """factor() re-runs both analysis and numeric factorization."""
        A_full, A_upper = A4
        A2_full = A_full.copy()
        A2_full[0, 0] += 5.0
        A2_upper = _spd_upper_csr(A2_full)

        solver4.factor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Matrix types
# ---------------------------------------------------------------------------

class TestMatrixTypes:
    def test_sym_posdef(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_upper = _spd_upper_csr(A_full)
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_sym_indef(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_INDEF)
        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_nonsym(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 3.0, 6.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()
        solver = PardisoSolver(A_csr, MTYPE_REAL_NONSYM)
        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_struct_sym(self):
        A_full = np.array([
            [4.0, 0.5, 0.0],
            [1.5, 5.0, 2.0],
            [0.0, 2.0, 6.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()
        solver = PardisoSolver(A_csr, MTYPE_REAL_STRUCT_SYM)
        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# iparm access
# ---------------------------------------------------------------------------

class TestIparm:
    def test_get_iparm(self, solver4):
        iparm = solver4.get_iparm()
        assert len(iparm) == 64
        assert iparm[0] == 1
        assert iparm[34] == 1

    def test_set_iparm(self, solver4):
        solver4.set_iparm(7, 2)
        assert solver4.get_iparm_value(7) == 2

    def test_set_iparm_all(self, solver4):
        iparm = np.zeros(64, dtype=np.int64)
        iparm[0] = 1
        iparm[34] = 1
        iparm[7] = 5
        solver4.set_iparm_all(iparm)
        assert solver4.get_iparm_value(7) == 5


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

class TestPerm:
    def test_set_and_clear_perm(self, solver4):
        assert not solver4.has_perm()
        solver4.set_perm(np.array([0, 1, 2, 3], dtype=np.int64))
        assert solver4.has_perm()
        solver4.clear_perm()
        assert not solver4.has_perm()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_n(self, solver4):
        assert solver4.n == 4

    def test_nnz(self, solver4, A4):
        _, A_upper = A4
        assert solver4.nnz == A_upper.nnz

    def test_mtype(self, solver4):
        assert solver4.mtype == MTYPE_REAL_SYM_POSDEF


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release(self, solver4):
        solver4.release()
        # After release, solve should fail
        with pytest.raises(RuntimeError, match="not factored"):
            solver4.solve(np.ones(4))


# ---------------------------------------------------------------------------
# Larger systems
# ---------------------------------------------------------------------------

class TestLarger:
    def test_random_spd_100(self):
        _, M_upper = _make_spd(100, density=0.05, seed=42)
        M_full = (M_upper + sp.triu(M_upper, k=1).T).toarray()

        solver = PardisoSolver(M_upper, MTYPE_REAL_SYM_POSDEF)
        rng = np.random.default_rng(42)
        b = rng.standard_normal(100)
        x = solver.solve(b)
        npt.assert_allclose(M_full @ x, b, atol=1e-8)

    def test_random_nonsym_200(self):
        rng = np.random.default_rng(7)
        n = 200
        A = sp.random(n, n, density=0.03, random_state=rng, format="csr")
        A = A + 5.0 * sp.eye(n, format="csr")
        A.sort_indices()

        solver = PardisoSolver(A, MTYPE_REAL_NONSYM)
        b = rng.standard_normal(n)
        x = solver.solve(b)
        npt.assert_allclose(A.toarray() @ x, b, atol=1e-8)


# ---------------------------------------------------------------------------
# Lifecycle: refactor loop (qtqp-style workflow)
# ---------------------------------------------------------------------------

class TestQtqpWorkflow:
    def test_indef_refactor_loop(self):
        """Simulate the qtqp IPM workflow: sym indef with scaling/matching."""
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()

        solver = PardisoSolver(
            A_upper, MTYPE_REAL_SYM_INDEF,
            iparms={9: 8, 10: 1, 12: 1, 23: 1},
        )

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-6)

        # Refactor with modified diagonal
        for delta in [0.5, 1.0, 2.0]:
            A2_full = A_full.copy()
            A2_full[0, 0] += delta
            A2_full[2, 2] += delta
            A2_upper = sp.csr_matrix(np.triu(A2_full))
            A2_upper.sort_indices()
            # These iparm settings invalidate symbolic analysis, so recovery
            # must go through factor(), not refactor().
            solver.factor(A2_upper.data.astype(np.float64))
            x = solver.solve(b)
            npt.assert_allclose(A2_full @ x, b, atol=1e-6)

    def test_full_matrix_construction_rejected_for_symmetric_type(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()

        with pytest.raises(ValueError, match="upper-triangular"):
            PardisoSolver(A_csr, MTYPE_REAL_SYM_INDEF, iparms={9: 8, 23: 1})

    def test_full_matrix_refactor_rejected(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_INDEF, iparms={9: 8, 23: 1})

        A2_full = A_full.copy()
        A2_full[0, 0] += 1.0
        A2_full[2, 2] += 1.0
        A2_csr = sp.csr_matrix(A2_full)
        A2_csr.sort_indices()
        with pytest.raises(ValueError, match="nnz"):
            solver.refactor(A2_csr.data.astype(np.float64))

    def test_full_matrix_factor_error_recovery_rejected(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()
        solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_INDEF, iparms={9: 8, 23: 1})

        solver.set_iparm(10, 1)
        solver.set_iparm(12, 1)
        A2_full = A_full.copy()
        A2_full[1, 1] = -5.0
        A2_csr = sp.csr_matrix(A2_full)
        A2_csr.sort_indices()
        with pytest.raises(ValueError, match="nnz"):
            solver.factor(A2_csr.data.astype(np.float64))

    def test_full_spd_matrix_rejected(self):
        A_full = np.array([
            [10.0, 1.0, 0.0, 0.0],
            [1.0, 8.0, 2.0, 0.0],
            [0.0, 2.0, 12.0, 1.0],
            [0.0, 0.0, 1.0, 6.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()

        with pytest.raises(ValueError, match="upper-triangular"):
            PardisoSolver(A_csr, MTYPE_REAL_SYM_POSDEF)

    def test_factor_for_error_recovery(self):
        """factor() can be used for error recovery (re-analyzes from scratch)."""
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()

        solver = PardisoSolver(
            A_upper, MTYPE_REAL_SYM_INDEF,
            iparms={10: 1, 12: 1},
        )

        # Simulate error recovery by calling factor()
        A2_full = A_full.copy()
        A2_full[1, 1] = -5.0
        A2_upper = sp.csr_matrix(np.triu(A2_full))
        A2_upper.sort_indices()
        solver.factor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-6)
