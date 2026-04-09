import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

import pymklpardiso


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spd_upper_csr(A_full):
    """Return upper-triangular CSR representation of a symmetric matrix."""
    return sp.csr_matrix(np.triu(A_full))


def _set_pattern_from_csr(solver, M):
    """Set the CSR sparsity pattern on a PardisoSolver."""
    M = sp.csr_matrix(M)
    M.sort_indices()
    solver.set_pattern(
        ia=M.indptr.astype(np.int64),
        ja=M.indices.astype(np.int64),
        n=M.shape[0],
    )
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
    """PardisoSolver factored for the 4x4 SPD fixture (upper triangle)."""
    A_full, A_upper = A4
    solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
    _set_pattern_from_csr(solver, A_upper)
    solver.factor(A_upper.data.astype(np.float64))
    return solver


# ---------------------------------------------------------------------------
# Basic solve
# ---------------------------------------------------------------------------

class TestSolve1D:
    def test_solve_1d(self, solver4, A4):
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A_full, b)
        npt.assert_allclose(x, x_ref, atol=1e-12)

    def test_residual_1d(self, solver4, A4):
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_solve_1d_integer_rhs(self, solver4, A4):
        """Integer RHS should be cast to float64 automatically."""
        A_full, _ = A4
        b = np.array([1, 2, 3, 4])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A_full, b.astype(float))
        npt.assert_allclose(x, x_ref, atol=1e-12)

    def test_multiple_solves_same_factor(self, solver4, A4):
        """Multiple solves without re-factoring should all be correct."""
        A_full, _ = A4
        for i in range(5):
            b = np.array([1.0 + i, 2.0, 3.0, 4.0 - i])
            x = solver4.solve(b)
            npt.assert_allclose(A_full @ x, b, atol=1e-12)


class TestSolve2D:
    def test_solve_2d(self, solver4, A4):
        A_full, _ = A4
        B = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        X = solver4.solve(B)
        X_ref = np.linalg.solve(A_full, B)
        npt.assert_allclose(X, X_ref, atol=1e-12)

    def test_solve_2d_f_contiguous_input(self, solver4, A4):
        """F-contiguous input should also work (no copy needed internally)."""
        A_full, _ = A4
        B = np.asfortranarray(
            np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        )
        X = solver4.solve(B)
        X_ref = np.linalg.solve(A_full, B)
        npt.assert_allclose(X, X_ref, atol=1e-12)

    def test_solve_2d_output_is_f_contiguous(self, solver4):
        B = np.ones((4, 3))
        X = solver4.solve(B)
        assert X.flags["F_CONTIGUOUS"]

    def test_residual_2d(self, solver4, A4):
        A_full, _ = A4
        B = np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        X = solver4.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)

    def test_solve_2d_single_rhs(self, solver4, A4):
        """2D array with a single RHS column."""
        A_full, _ = A4
        B = np.array([[1.0], [2.0], [3.0], [4.0]])
        X = solver4.solve(B)
        assert X.shape == (4, 1)
        x_ref = np.linalg.solve(A_full, B)
        npt.assert_allclose(X, x_ref, atol=1e-12)

    def test_solve_2d_many_rhs(self, solver4, A4):
        """2D solve with many RHS columns."""
        A_full, _ = A4
        rng = np.random.default_rng(123)
        B = rng.standard_normal((4, 20))
        X = solver4.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-10)


# ---------------------------------------------------------------------------
# solve_into
# ---------------------------------------------------------------------------

class TestSolveInto:
    def test_solve_into_1d(self, solver4, A4):
        A_full, _ = A4
        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(4)
        solver4.solve_into(b, x)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_solve_into_2d(self, solver4, A4):
        A_full, _ = A4
        B = np.asfortranarray(
            np.array([[1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]])
        )
        X = np.asfortranarray(np.zeros((4, 2)))
        solver4.solve_into(B, X)
        npt.assert_allclose(X, np.linalg.solve(A_full, B), atol=1e-12)

    def test_solve_into_rejects_c_contiguous_2d(self, solver4):
        B = np.ones((4, 2))  # C-contiguous
        X = np.zeros((4, 2))  # C-contiguous
        with pytest.raises(ValueError, match="Fortran-contiguous"):
            solver4.solve_into(B, X)

    def test_solve_into_rejects_rank_mismatch(self, solver4):
        """b and x must have the same rank."""
        b = np.ones(4)
        x = np.asfortranarray(np.zeros((4, 1)))
        with pytest.raises(ValueError, match="rank"):
            solver4.solve_into(b, x)

    def test_solve_into_rejects_wrong_length_1d(self, solver4):
        b = np.ones(3)
        x = np.zeros(3)
        with pytest.raises(ValueError, match="length n"):
            solver4.solve_into(b, x)

    def test_solve_into_rejects_wrong_shape_2d(self, solver4):
        B = np.asfortranarray(np.ones((3, 2)))
        X = np.asfortranarray(np.zeros((3, 2)))
        with pytest.raises(ValueError, match="shape"):
            solver4.solve_into(B, X)

    def test_solve_into_rejects_nrhs_mismatch(self, solver4):
        B = np.asfortranarray(np.ones((4, 2)))
        X = np.asfortranarray(np.zeros((4, 3)))
        with pytest.raises(ValueError, match="right-hand sides"):
            solver4.solve_into(B, X)

    def test_solve_into_rejects_3d(self, solver4):
        b = np.ones((4, 2, 2))
        x = np.zeros((4, 2, 2))
        with pytest.raises(ValueError):
            solver4.solve_into(b, x)

    def test_solve_into_before_factor(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="not factored"):
            solver.solve_into(np.ones(4), np.zeros(4))


# ---------------------------------------------------------------------------
# Refactor (same pattern, new values)
# ---------------------------------------------------------------------------

class TestRefactor:
    def test_refactor_correctness(self, solver4, A4):
        A_full, A_upper = A4
        A2_full = A_full.copy()
        A2_full[0, 0] += 2.0
        A2_full[1, 1] += 1.0
        A2_full[2, 2] += 3.0
        A2_full[3, 3] += 0.5

        A2_upper = _spd_upper_csr(A2_full)
        solver4.set_values(A2_upper.data.astype(np.float64))
        solver4.refactor()

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver4.solve(b)
        x_ref = np.linalg.solve(A2_full, b)
        npt.assert_allclose(x, x_ref, atol=1e-12)

    def test_repeated_refactor(self, A4):
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0, 4.0])
        for i in range(10):
            A_mod = A_full.copy()
            A_mod[0, 0] += 0.5 * i
            A_mod[2, 2] += 0.3 * i
            A_csr = _spd_upper_csr(A_mod)
            solver.set_values(A_csr.data.astype(np.float64))
            solver.refactor()
            x = solver.solve(b)
            npt.assert_allclose(A_mod @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Matrix types
# ---------------------------------------------------------------------------

class TestMatrixTypes:
    def test_sym_posdef(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_upper = _spd_upper_csr(A_full)

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

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

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_INDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

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

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_struct_sym(self):
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, 5.0, 2.0],
            [0.0, 2.0, 6.0],
        ])
        # Structurally symmetric but numerically not (pass full matrix).
        A_full_unsym = A_full.copy()
        A_full_unsym[0, 1] = 0.5  # break numeric symmetry
        A_full_unsym[1, 0] = 1.5

        A_csr = sp.csr_matrix(A_full_unsym)
        A_csr.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_STRUCT_SYM)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full_unsym @ x, b, atol=1e-12)

    def test_nonsym_2d_solve(self):
        """Nonsymmetric matrix with multiple RHS."""
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [2.0, 5.0, 1.0],
            [0.0, 3.0, 6.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))

        B = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X = solver.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)

    def test_sym_indef_2d_solve(self):
        """Symmetric indefinite matrix with multiple RHS."""
        A_full = np.array([
            [4.0, 1.0, 0.0],
            [1.0, -3.0, 2.0],
            [0.0, 2.0, 5.0],
        ])
        A_upper = sp.csr_matrix(np.triu(A_full))
        A_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_INDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        B = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        X = solver.solve(B)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)


# ---------------------------------------------------------------------------
# iparm access
# ---------------------------------------------------------------------------

class TestIparm:
    def test_get_iparm(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = solver.get_iparm()
        assert len(iparm) == 64
        assert iparm[0] == 1   # user-supplied iparm
        assert iparm[34] == 1  # zero-based indexing

    def test_set_iparm(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_iparm(7, 2)
        assert solver.get_iparm_value(7) == 2

    def test_set_iparm_rejects_iparm0(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="iparm\\[0\\]"):
            solver.set_iparm(0, 0)

    def test_set_iparm_rejects_iparm34(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="iparm\\[34\\]"):
            solver.set_iparm(34, 0)

    def test_set_iparm_allows_iparm0_equals_1(self):
        """Setting iparm[0] to 1 (the locked value) should succeed."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_iparm(0, 1)
        assert solver.get_iparm_value(0) == 1

    def test_set_iparm_allows_iparm34_equals_1(self):
        """Setting iparm[34] to 1 (the locked value) should succeed."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_iparm(34, 1)
        assert solver.get_iparm_value(34) == 1

    def test_set_iparm_all(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = np.zeros(64, dtype=np.int64)
        iparm[0] = 1
        iparm[34] = 1
        iparm[7] = 5
        solver.set_iparm_all(iparm)
        assert solver.get_iparm_value(7) == 5

    def test_set_iparm_index_out_of_range(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="\\[0, 63\\]"):
            solver.set_iparm(64, 0)

    def test_set_iparm_index_negative(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="\\[0, 63\\]"):
            solver.set_iparm(-1, 0)

    def test_set_iparm_all_wrong_length(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="length 64"):
            solver.set_iparm_all(np.zeros(32, dtype=np.int64))

    def test_set_iparm_all_rejects_iparm0_not_1(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = np.zeros(64, dtype=np.int64)
        iparm[34] = 1
        # iparm[0] is 0, not 1
        with pytest.raises(ValueError, match="iparm\\[0\\]"):
            solver.set_iparm_all(iparm)

    def test_set_iparm_all_rejects_iparm34_not_1(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = np.zeros(64, dtype=np.int64)
        iparm[0] = 1
        # iparm[34] is 0, not 1
        with pytest.raises(ValueError, match="iparm\\[34\\]"):
            solver.set_iparm_all(iparm)

    def test_set_iparm_all_rejects_non_1d(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = np.zeros((8, 8), dtype=np.int64)
        with pytest.raises(ValueError, match="1D"):
            solver.set_iparm_all(iparm)

    def test_set_iparm_all_coerces_int32(self):
        """int32 input should be coerced to int64."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        iparm = np.zeros(64, dtype=np.int32)
        iparm[0] = 1
        iparm[34] = 1
        iparm[7] = 3
        solver.set_iparm_all(iparm)
        assert solver.get_iparm_value(7) == 3

    def test_get_iparm_value_out_of_range(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="\\[0, 63\\]"):
            solver.get_iparm_value(64)


# ---------------------------------------------------------------------------
# Pattern and value management
# ---------------------------------------------------------------------------

class TestPatternValues:
    def test_set_pattern(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        assert solver.n() == 4

    def test_set_values_wrong_length(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        with pytest.raises(ValueError, match="nnz"):
            solver.set_values(np.array([1.0, 2.0]))

    def test_set_pattern_rejects_negative_n(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="positive"):
            solver.set_pattern(
                ia=np.array([0, 1], dtype=np.int64),
                ja=np.array([0], dtype=np.int64),
                n=-1,
            )

    def test_set_pattern_rejects_zero_n(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="positive"):
            solver.set_pattern(
                ia=np.array([0], dtype=np.int64),
                ja=np.array([], dtype=np.int64),
                n=0,
            )

    def test_set_pattern_ia_wrong_length(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="ia"):
            solver.set_pattern(
                ia=np.array([0, 1, 2], dtype=np.int64),  # length 3, but n=1 expects 2
                ja=np.array([0], dtype=np.int64),
                n=1,
            )

    def test_set_pattern_unsorted_rejected(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="sorted"):
            solver.set_pattern(
                ia=np.array([0, 2, 2], dtype=np.int64),
                ja=np.array([1, 0], dtype=np.int64),  # unsorted within row 0
                n=2,
            )

    def test_set_pattern_unsorted_allowed_if_check_disabled(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        # Should not raise when check_sorted=False
        solver.set_pattern(
            ia=np.array([0, 2, 2], dtype=np.int64),
            ja=np.array([1, 0], dtype=np.int64),
            n=2,
            check_sorted=False,
        )

    def test_set_pattern_negative_nnz(self):
        """ia[n] < 0 should be rejected."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="nonnegative"):
            solver.set_pattern(
                ia=np.array([0, -1], dtype=np.int64),
                ja=np.array([], dtype=np.int64),
                n=1,
            )

    def test_set_pattern_ja_size_mismatch(self):
        """ja length != ia[n] should be rejected."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="ja"):
            solver.set_pattern(
                ia=np.array([0, 2], dtype=np.int64),
                ja=np.array([0], dtype=np.int64),  # length 1, but ia[n]=2
                n=1,
            )

    def test_set_pattern_ia_not_nondecreasing(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="nondecreasing"):
            solver.set_pattern(
                ia=np.array([0, 2, 1], dtype=np.int64),
                ja=np.array([0, 1], dtype=np.int64),
                n=2,
            )

    def test_set_pattern_ja_out_of_range(self):
        """Column index >= n should be rejected."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="out-of-range"):
            solver.set_pattern(
                ia=np.array([0, 1], dtype=np.int64),
                ja=np.array([5], dtype=np.int64),  # 5 >= n=1
                n=1,
            )

    def test_set_pattern_ja_negative_index(self):
        """Negative column index should be rejected."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="out-of-range"):
            solver.set_pattern(
                ia=np.array([0, 1], dtype=np.int64),
                ja=np.array([-1], dtype=np.int64),
                n=1,
            )

    def test_set_pattern_ia_non_1d(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="1D"):
            solver.set_pattern(
                ia=np.array([[0, 1]], dtype=np.int64),
                ja=np.array([0], dtype=np.int64),
                n=1,
            )

    def test_set_pattern_ja_non_1d(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(ValueError, match="1D"):
            solver.set_pattern(
                ia=np.array([0, 1], dtype=np.int64),
                ja=np.array([[0]], dtype=np.int64),
                n=1,
            )

    def test_set_values_non_1d(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        with pytest.raises(ValueError, match="1D"):
            solver.set_values(np.array([[1.0]]))

    def test_set_pattern_coerces_int32(self):
        """int32 ia/ja should be coerced to int64."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int32),
            ja=np.array([0], dtype=np.int32),
            n=1,
        )
        assert solver.n() == 1

    def test_set_pattern_empty_rows(self):
        """Matrix with empty rows (no nonzeros in some rows)."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        # 3x3 matrix: row 0 has element (0,0), row 1 is empty, row 2 has (2,2)
        solver.set_pattern(
            ia=np.array([0, 1, 1, 2], dtype=np.int64),
            ja=np.array([0, 2], dtype=np.int64),
            n=3,
        )
        assert solver.n() == 3

    def test_pattern_change_after_factor(self, A4):
        """Setting a new pattern after factorization should work."""
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        # Now set a different pattern
        A2_full = np.array([[5.0, 1.0], [1.0, 4.0]])
        A2_upper = _spd_upper_csr(A2_full)
        _set_pattern_from_csr(solver, A2_upper)
        solver.factor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------

class TestPerm:
    def test_set_and_clear_perm(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        assert not solver.has_perm()
        solver.set_perm(np.array([0, 1, 2, 3], dtype=np.int64))
        assert solver.has_perm()
        solver.clear_perm()
        assert not solver.has_perm()

    def test_set_perm_wrong_length(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        with pytest.raises(ValueError, match="length n"):
            solver.set_perm(np.array([0, 1], dtype=np.int64))

    def test_set_perm_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="set_pattern"):
            solver.set_perm(np.array([0, 1], dtype=np.int64))

    def test_set_perm_non_1d(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        with pytest.raises(ValueError, match="1D"):
            solver.set_perm(np.array([[0, 1, 2, 3]], dtype=np.int64))

    def test_perm_coerces_int32(self, A4):
        """int32 perm should be coerced to int64."""
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_perm(np.array([0, 1, 2, 3], dtype=np.int32))
        assert solver.has_perm()


# ---------------------------------------------------------------------------
# State machine errors
# ---------------------------------------------------------------------------

class TestStateErrors:
    def test_solve_before_factor(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="not factored"):
            solver.solve(np.array([1.0]))

    def test_factor_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="set_pattern"):
            solver.factor(np.array([1.0]))

    def test_refactor_before_values(self):
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_upper = _spd_upper_csr(A_full)
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        with pytest.raises(RuntimeError, match="not set"):
            solver.refactor()

    def test_set_values_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="set_pattern"):
            solver.set_values(np.array([1.0]))

    def test_analyze_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="set_pattern"):
            solver.analyze()

    def test_run_phase_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        with pytest.raises(RuntimeError, match="set_pattern"):
            solver.run_phase(11)

    def test_run_phase_22_before_values(self):
        """Phase 22 requires values to be set."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        with pytest.raises(RuntimeError, match="values"):
            solver.run_phase(22)

    def test_run_phase_12_before_values(self):
        """Phase 12 requires values to be set."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        with pytest.raises(RuntimeError, match="values"):
            solver.run_phase(12)

    def test_run_phase_23_before_values(self):
        """Phase 23 requires values to be set."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        with pytest.raises(RuntimeError, match="values"):
            solver.run_phase(23)

    def test_run_phase_33_before_factor(self):
        """Phase 33 (solve) requires prior factorization."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        solver.set_values(np.array([1.0]))
        with pytest.raises(RuntimeError, match="factorization"):
            solver.run_phase(33)


# ---------------------------------------------------------------------------
# Validation on solve
# ---------------------------------------------------------------------------

class TestSolveValidation:
    def test_rejects_wrong_rhs_length(self, solver4):
        with pytest.raises(ValueError, match="length n"):
            solver4.solve(np.array([1.0, 2.0]))

    def test_rejects_3d_rhs(self, solver4):
        with pytest.raises(ValueError):
            solver4.solve(np.ones((4, 2, 2)))

    def test_rejects_wrong_shape_2d(self, solver4):
        with pytest.raises(ValueError, match="shape"):
            solver4.solve(np.ones((3, 2)))


# ---------------------------------------------------------------------------
# Larger system
# ---------------------------------------------------------------------------

class TestLarger:
    def test_random_spd_100(self):
        rng = np.random.default_rng(42)
        n = 100
        density = 0.05
        R = sp.random(n, n, density=density, random_state=rng, format="csr")
        M = (R.T @ R + sp.eye(n)).tocsr()
        M_upper = sp.triu(M, format="csr")
        M_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, M_upper)
        solver.factor(M_upper.data.astype(np.float64))

        b = rng.standard_normal(n)
        x = solver.solve(b)
        npt.assert_allclose(M.toarray() @ x, b, atol=1e-8)

    def test_random_spd_multi_rhs(self):
        rng = np.random.default_rng(99)
        n = 50
        R = sp.random(n, n, density=0.1, random_state=rng, format="csr")
        M = (R.T @ R + sp.eye(n)).tocsr()
        M_upper = sp.triu(M, format="csr")
        M_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, M_upper)
        solver.factor(M_upper.data.astype(np.float64))

        B = rng.standard_normal((n, 5))
        X = solver.solve(B)
        npt.assert_allclose(M.toarray() @ X, B, atol=1e-8)

    def test_random_nonsym_200(self):
        rng = np.random.default_rng(7)
        n = 200
        A = sp.random(n, n, density=0.03, random_state=rng, format="csr")
        A = A + 5.0 * sp.eye(n, format="csr")  # make diagonally dominant
        A.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A)
        solver.factor(A.data.astype(np.float64))

        b = rng.standard_normal(n)
        x = solver.solve(b)
        npt.assert_allclose(A.toarray() @ x, b, atol=1e-8)

    def test_random_sym_indef_50(self):
        """Random symmetric indefinite system."""
        rng = np.random.default_rng(55)
        n = 50
        R = sp.random(n, n, density=0.1, random_state=rng, format="csr")
        M = (R + R.T).tocsr()
        # Add diagonal to make it non-singular (but not positive definite)
        diag_vals = rng.standard_normal(n) * 3
        M = M + sp.diags(diag_vals, format="csr")
        M_upper = sp.triu(M, format="csr")
        M_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_INDEF)
        _set_pattern_from_csr(solver, M_upper)
        solver.factor(M_upper.data.astype(np.float64))

        b = rng.standard_normal(n)
        x = solver.solve(b)
        npt.assert_allclose(M.toarray() @ x, b, atol=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_1x1_matrix(self):
        A_csr = sp.csr_matrix(np.array([[3.0]]))
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))
        x = solver.solve(np.array([6.0]))
        npt.assert_allclose(x, [2.0], atol=1e-14)

    def test_diagonal_matrix(self):
        d = np.array([2.0, 5.0, 0.5, 10.0])
        A = sp.diags(d, format="csr")
        A.sort_indices()
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A)
        solver.factor(A.data.astype(np.float64))
        b = np.array([1.0, 1.0, 1.0, 1.0])
        x = solver.solve(b)
        npt.assert_allclose(x, 1.0 / d, atol=1e-14)

    def test_float32_values_coerced(self):
        """float32 values should be cast to float64 via forcecast."""
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_upper = _spd_upper_csr(A_full)
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float32))  # float32 input
        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b), atol=1e-12)

    def test_float32_rhs_coerced(self):
        """float32 RHS should be cast to float64 via forcecast."""
        A_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A_upper = _spd_upper_csr(A_full)
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))
        b = np.array([1.0, 2.0], dtype=np.float32)
        x = solver.solve(b)
        npt.assert_allclose(x, np.linalg.solve(A_full, b.astype(np.float64)), atol=1e-12)

    def test_dense_small_matrix(self):
        """Fully dense small matrix."""
        A_full = np.array([
            [10.0, 1.0, 2.0],
            [3.0, 8.0, 1.0],
            [2.0, 1.0, 9.0],
        ])
        A_csr = sp.csr_matrix(A_full)
        A_csr.sort_indices()
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))
        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_tridiagonal_matrix(self):
        """Tridiagonal symmetric positive definite matrix."""
        n = 20
        diags = [np.ones(n - 1) * (-0.5), np.ones(n) * 3.0, np.ones(n - 1) * (-0.5)]
        A = sp.diags(diags, offsets=[-1, 0, 1], format="csr")
        A_upper = sp.triu(A, format="csr")
        A_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        b = np.ones(n)
        x = solver.solve(b)
        npt.assert_allclose(A.toarray() @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Reset and release
# ---------------------------------------------------------------------------

class TestResetRelease:
    def test_release_and_refactor(self, A4):
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x1 = solver.solve(b)

        solver.release()
        # After release, factor again with same data
        solver.factor(A_upper.data.astype(np.float64))
        x2 = solver.solve(b)
        npt.assert_allclose(x1, x2, atol=1e-14)

    def test_reset(self, A4):
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        solver.reset()
        assert solver.n() == 0

        # Can set a new pattern and factor after reset
        A2_full = np.array([[5.0, 1.0], [1.0, 4.0]])
        A2_upper = _spd_upper_csr(A2_full)
        _set_pattern_from_csr(solver, A2_upper)
        solver.factor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(A2_full @ x, b, atol=1e-12)

    def test_release_idempotent(self):
        """Calling release() multiple times should not crash."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.release()
        solver.release()

    def test_release_after_factor(self, A4):
        """release() then solve should raise."""
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))
        solver.release()
        with pytest.raises(RuntimeError, match="not factored"):
            solver.solve(np.ones(4))

    def test_set_pattern_after_release(self, A4):
        """Can set a new pattern after release (without reset)."""
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))
        solver.release()

        # Set new pattern and factor
        A2_full = np.array([[5.0]])
        A2_csr = sp.csr_matrix(A2_full)
        _set_pattern_from_csr(solver, A2_csr)
        solver.factor(A2_csr.data.astype(np.float64))
        x = solver.solve(np.array([10.0]))
        npt.assert_allclose(x, [2.0], atol=1e-14)

    def test_reset_clears_perm(self, A4):
        """reset() should clear the permutation."""
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_perm(np.array([0, 1, 2, 3], dtype=np.int64))
        assert solver.has_perm()
        solver.reset()
        assert not solver.has_perm()


# ---------------------------------------------------------------------------
# run_phase
# ---------------------------------------------------------------------------

class TestRunPhase:
    def test_analyze_factor_solve_phases(self, A4):
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_values(A_upper.data.astype(np.float64))

        # Phase 11: symbolic analysis
        solver.run_phase(11)
        # Phase 22: numerical factorization
        solver.run_phase(22)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_release_via_run_phase(self, solver4):
        solver4.run_phase(-1)
        # After release, solve should fail
        with pytest.raises(RuntimeError, match="not factored"):
            solver4.solve(np.array([1.0, 2.0, 3.0, 4.0]))

    def test_combined_phase_12(self, A4):
        """Phase 12 = analyze + factor in one call."""
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_values(A_upper.data.astype(np.float64))
        solver.run_phase(12)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_combined_phase_23(self, A4):
        """Phase 23 = factor + solve in one call (via run_phase_into)."""
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_values(A_upper.data.astype(np.float64))
        solver.run_phase(11)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(4)
        solver.run_phase_into(23, b, x)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_explicit_analyze(self, A4):
        """analyze() followed by refactor() should work."""
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.set_values(A_upper.data.astype(np.float64))
        solver.analyze()
        solver.refactor()

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)


# ---------------------------------------------------------------------------
# run_phase_into
# ---------------------------------------------------------------------------

class TestRunPhaseInto:
    def test_solve_phase_into_1d(self, A4):
        """Use run_phase_into for phase 33 (solve) with 1D arrays."""
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.zeros(4)
        solver.run_phase_into(33, b, x)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

    def test_solve_phase_into_2d(self, A4):
        """Use run_phase_into for phase 33 with 2D F-contiguous arrays."""
        A_full, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        B = np.asfortranarray(np.array([
            [1.0, 2.0], [0.0, 1.0], [3.0, 4.0], [1.0, 0.0]
        ]))
        X = np.asfortranarray(np.zeros((4, 2)))
        solver.run_phase_into(33, B, X)
        npt.assert_allclose(A_full @ X, B, atol=1e-12)

    def test_run_phase_into_rejects_phase_minus_1(self, solver4):
        """Phase -1 should be rejected (use release() instead)."""
        b = np.ones(4)
        x = np.zeros(4)
        with pytest.raises(ValueError, match="release"):
            solver4.run_phase_into(-1, b, x)

    def test_run_phase_into_rejects_c_contiguous_2d(self, solver4):
        B = np.ones((4, 2))
        X = np.zeros((4, 2))
        with pytest.raises(ValueError, match="Fortran-contiguous"):
            solver4.run_phase_into(33, B, X)

    def test_run_phase_into_rejects_rank_mismatch(self, solver4):
        b = np.ones(4)
        x = np.asfortranarray(np.zeros((4, 1)))
        with pytest.raises(ValueError, match="rank"):
            solver4.run_phase_into(33, b, x)

    def test_run_phase_into_rejects_3d(self, solver4):
        b = np.ones((4, 2, 2))
        x = np.zeros((4, 2, 2))
        with pytest.raises(ValueError):
            solver4.run_phase_into(33, b, x)

    def test_run_phase_into_wrong_length_1d(self, solver4):
        b = np.ones(3)
        x = np.zeros(3)
        with pytest.raises(ValueError, match="length n"):
            solver4.run_phase_into(33, b, x)

    def test_run_phase_into_phase_33_before_factor(self):
        """run_phase_into(33, ...) should require factorization."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        solver.set_values(np.array([1.0]))
        b = np.array([1.0])
        x = np.zeros(1)
        with pytest.raises(RuntimeError, match="factorization"):
            solver.run_phase_into(33, b, x)


# ---------------------------------------------------------------------------
# msglvl
# ---------------------------------------------------------------------------

class TestMsglvl:
    def test_set_msglvl(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF, msglvl=1)
        solver.set_msglvl(0)
        # No assertion beyond not crashing; msglvl controls printing only.

    def test_default_msglvl(self):
        """Default msglvl should be 0."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        # Just verifying no error; msglvl is not exposed as a getter.


# ---------------------------------------------------------------------------
# mtype accessor
# ---------------------------------------------------------------------------

class TestMtype:
    def test_mtype_accessor(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        assert solver.mtype() == pymklpardiso.MTYPE_REAL_SYM_POSDEF

    def test_mtype_values(self):
        assert pymklpardiso.MTYPE_REAL_STRUCT_SYM == 1
        assert pymklpardiso.MTYPE_REAL_SYM_INDEF == -2
        assert pymklpardiso.MTYPE_REAL_SYM_POSDEF == 2
        assert pymklpardiso.MTYPE_REAL_NONSYM == 11

    def test_mtype_accessor_all_types(self):
        for mtype in [
            pymklpardiso.MTYPE_REAL_STRUCT_SYM,
            pymklpardiso.MTYPE_REAL_SYM_INDEF,
            pymklpardiso.MTYPE_REAL_SYM_POSDEF,
            pymklpardiso.MTYPE_REAL_NONSYM,
        ]:
            solver = pymklpardiso.PardisoSolver(mtype)
            assert solver.mtype() == mtype


# ---------------------------------------------------------------------------
# n() accessor
# ---------------------------------------------------------------------------

class TestNAccessor:
    def test_n_before_pattern(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        assert solver.n() == 0

    def test_n_after_pattern(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        assert solver.n() == 4

    def test_n_after_reset(self, A4):
        _, A_upper = A4
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.reset()
        assert solver.n() == 0


# ---------------------------------------------------------------------------
# PARDISO error handling (singular / near-singular matrices)
# ---------------------------------------------------------------------------

class TestPardisoErrors:
    def test_singular_matrix_detection(self):
        """Factoring a singular matrix should raise a RuntimeError."""
        # Zero diagonal = singular
        A = sp.csr_matrix(np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ]))
        A_upper = sp.triu(A, format="csr")
        A_upper.sort_indices()

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_INDEF)
        _set_pattern_from_csr(solver, A_upper)
        with pytest.raises(RuntimeError, match="PARDISO"):
            solver.factor(A_upper.data.astype(np.float64))

    def test_zero_pivot_nonsym(self):
        """Nonsymmetric singular matrix should raise."""
        A = sp.csr_matrix(np.array([
            [1.0, 2.0],
            [0.5, 1.0],
        ]))
        A.sort_indices()
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A)
        with pytest.raises(RuntimeError, match="PARDISO"):
            solver.factor(A.data.astype(np.float64))


# ---------------------------------------------------------------------------
# Perm with iparm[30]/iparm[35] enforcement
# ---------------------------------------------------------------------------

class TestPermIparmEnforcement:
    def test_iparm30_requires_perm(self):
        """iparm[30]=1 requires perm to be set."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        solver.set_values(np.array([1.0]))
        solver.set_iparm(30, 1)
        with pytest.raises(RuntimeError, match="perm"):
            solver.run_phase(11)

    def test_iparm35_requires_perm(self):
        """iparm[35]=1 requires perm to be set."""
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        solver.set_pattern(
            ia=np.array([0, 1], dtype=np.int64),
            ja=np.array([0], dtype=np.int64),
            n=1,
        )
        solver.set_values(np.array([1.0]))
        solver.set_iparm(35, 1)
        with pytest.raises(RuntimeError, match="perm"):
            solver.run_phase(11)


# ---------------------------------------------------------------------------
# Full lifecycle / integration
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_full_spd_lifecycle(self):
        """Complete lifecycle: create, pattern, factor, solve, refactor, solve, release."""
        A_full = np.array([
            [10.0, 1.0, 0.0],
            [1.0, 8.0, 2.0],
            [0.0, 2.0, 12.0],
        ])
        A_upper = _spd_upper_csr(A_full)

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        _set_pattern_from_csr(solver, A_upper)
        solver.factor(A_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0, 3.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

        # Modify diagonal and refactor
        A2_full = A_full.copy()
        A2_full[0, 0] = 15.0
        A2_upper = _spd_upper_csr(A2_full)
        solver.set_values(A2_upper.data.astype(np.float64))
        solver.refactor()

        x2 = solver.solve(b)
        npt.assert_allclose(A2_full @ x2, b, atol=1e-12)

        solver.release()
        with pytest.raises(RuntimeError, match="not factored"):
            solver.solve(b)

    def test_full_nonsym_lifecycle(self):
        """Complete lifecycle for nonsymmetric matrix."""
        A_full = np.array([
            [5.0, 1.0],
            [2.0, 4.0],
        ])
        A_csr = sp.csr_matrix(A_full)

        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_NONSYM)
        _set_pattern_from_csr(solver, A_csr)
        solver.factor(A_csr.data.astype(np.float64))

        b = np.array([1.0, 2.0])
        x = solver.solve(b)
        npt.assert_allclose(A_full @ x, b, atol=1e-12)

        # solve_into
        x2 = np.zeros(2)
        solver.solve_into(b, x2)
        npt.assert_allclose(x, x2, atol=1e-14)

        # reset and re-use
        solver.reset()
        assert solver.n() == 0

        A3_full = np.array([
            [3.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [2.0, 0.0, 5.0],
        ])
        A3_csr = sp.csr_matrix(A3_full)
        _set_pattern_from_csr(solver, A3_csr)
        solver.factor(A3_csr.data.astype(np.float64))

        b3 = np.array([1.0, 2.0, 3.0])
        x3 = solver.solve(b3)
        npt.assert_allclose(A3_full @ x3, b3, atol=1e-12)

    def test_multiple_solvers(self):
        """Multiple solver instances should not interfere."""
        A1_full = np.array([[4.0, 1.0], [1.0, 3.0]])
        A1_upper = _spd_upper_csr(A1_full)

        A2_full = np.array([[6.0, 2.0], [2.0, 5.0]])
        A2_upper = _spd_upper_csr(A2_full)

        s1 = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)
        s2 = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF)

        _set_pattern_from_csr(s1, A1_upper)
        s1.factor(A1_upper.data.astype(np.float64))

        _set_pattern_from_csr(s2, A2_upper)
        s2.factor(A2_upper.data.astype(np.float64))

        b = np.array([1.0, 2.0])
        x1 = s1.solve(b)
        x2 = s2.solve(b)

        npt.assert_allclose(A1_full @ x1, b, atol=1e-12)
        npt.assert_allclose(A2_full @ x2, b, atol=1e-12)
        # Solutions should be different
        assert not np.allclose(x1, x2)
