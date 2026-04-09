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


# ---------------------------------------------------------------------------
# msglvl
# ---------------------------------------------------------------------------

class TestMsglvl:
    def test_set_msglvl(self):
        solver = pymklpardiso.PardisoSolver(pymklpardiso.MTYPE_REAL_SYM_POSDEF, msglvl=1)
        solver.set_msglvl(0)
        # No assertion beyond not crashing; msglvl controls printing only.


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
