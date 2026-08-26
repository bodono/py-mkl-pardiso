"""Complex-number coverage for the low- and high-level PARDISO wrappers."""

import numpy as np
import numpy.testing as npt
import pytest
import scipy.sparse as sp

import pymklpardiso._mkl_pardiso as low_level
from pymklpardiso import (
    MTYPE_COMPLEX_HERM_INDEF,
    MTYPE_COMPLEX_HERM_POSDEF,
    MTYPE_COMPLEX_NONSYM,
    MTYPE_COMPLEX_STRUCT_SYM,
    MTYPE_COMPLEX_SYM,
    MTYPE_REAL_NONSYM,
    PardisoSolver,
)


_COMPLEX_CASES = [
    pytest.param(
        MTYPE_COMPLEX_STRUCT_SYM,
        np.array([
            [5.0 + 1.0j, 1.0 + 2.0j, 0.0],
            [2.0 - 1.0j, 4.0 - 1.0j, 0.5j],
            [0.0, 1.0 + 0.25j, 3.0 + 0.5j],
        ]),
        False,
        id="structurally-symmetric",
    ),
    pytest.param(
        MTYPE_COMPLEX_HERM_POSDEF,
        np.array([
            [6.0, 1.0 + 2.0j, 0.5 - 0.25j],
            [1.0 - 2.0j, 7.0, 0.75 + 1.0j],
            [0.5 + 0.25j, 0.75 - 1.0j, 5.0],
        ]),
        True,
        id="hermitian-positive-definite",
    ),
    pytest.param(
        MTYPE_COMPLEX_HERM_INDEF,
        np.array([
            [4.0, 1.0 + 0.5j, 0.0],
            [1.0 - 0.5j, -3.0, 0.75 - 1.0j],
            [0.0, 0.75 + 1.0j, 2.0],
        ]),
        True,
        id="hermitian-indefinite",
    ),
    pytest.param(
        MTYPE_COMPLEX_SYM,
        np.array([
            [4.0 + 1.0j, 1.0 + 2.0j, 0.0],
            [1.0 + 2.0j, 3.0 - 0.5j, 0.5 - 0.75j],
            [0.0, 0.5 - 0.75j, 5.0 + 0.25j],
        ]),
        True,
        id="symmetric",
    ),
    pytest.param(
        MTYPE_COMPLEX_NONSYM,
        np.array([
            [5.0 + 1.0j, 1.0 - 2.0j, 0.0],
            [2.0 + 0.5j, 4.0 - 0.25j, 1.0j],
            [0.5 - 1.0j, 0.0, 3.0 + 0.75j],
        ]),
        False,
        id="nonsymmetric",
    ),
]


def _as_pardiso_csr(A, upper_only):
    matrix = sp.csr_matrix(np.triu(A) if upper_only else A)
    matrix.sort_indices()
    return matrix


def _set_pattern(solver, matrix):
    solver.set_pattern(
        ia=matrix.indptr.astype(np.int64),
        ja=matrix.indices.astype(np.int64),
        n=matrix.shape[0],
    )


@pytest.mark.parametrize("mtype,A,upper_only", _COMPLEX_CASES)
def test_low_level_supports_all_complex_matrix_types(mtype, A, upper_only):
    matrix = _as_pardiso_csr(A, upper_only)
    solver = low_level.PardisoSolver(mtype)
    _set_pattern(solver, matrix)
    solver.factor(matrix.data.astype(np.complex128))

    b = np.array([1.0 + 0.5j, -2.0 + 1.0j, 3.0 - 0.75j])
    x = solver.solve(b)

    assert x.dtype == np.complex128
    npt.assert_allclose(A @ x, b, atol=1e-12)


@pytest.mark.parametrize("mtype,A,upper_only", _COMPLEX_CASES)
def test_high_level_supports_all_complex_matrix_types(mtype, A, upper_only):
    matrix = _as_pardiso_csr(A, upper_only)
    solver = PardisoSolver(matrix, mtype)
    b = np.array([1.0 - 0.5j, 2.0 + 1.0j, -1.0 + 0.25j])

    x = solver.solve(b)

    assert x.dtype == np.complex128
    npt.assert_allclose(A @ x, b, atol=1e-12)


def test_complex_solve_promotes_real_rhs():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)
    b = np.array([1.0, 2.0, 3.0])

    x = solver.solve(b)

    assert x.dtype == np.complex128
    npt.assert_allclose(A @ x, b, atol=1e-12)


def test_complex_multiple_rhs_returns_fortran_contiguous_output():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)
    B = np.array([
        [1.0 + 0.5j, 2.0 - 1.0j],
        [0.0 + 1.0j, 1.0 + 0.25j],
        [3.0 - 0.5j, -1.0 + 2.0j],
    ])

    X = solver.solve(B)

    assert X.dtype == np.complex128
    assert X.flags["F_CONTIGUOUS"]
    npt.assert_allclose(A @ X, B, atol=1e-12)


def test_complex_solve_into_multiple_rhs():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)
    B = np.asfortranarray(np.array([
        [1.0 + 0.5j, 2.0 - 1.0j],
        [0.0 + 1.0j, 1.0 + 0.25j],
        [3.0 - 0.5j, -1.0 + 2.0j],
    ]))
    X = np.asfortranarray(np.zeros(B.shape, dtype=np.complex128))

    solver.solve_into(B, X)

    npt.assert_allclose(A @ X, B, atol=1e-12)


def test_complex_solve_into_promotes_real_fortran_contiguous_rhs():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)
    B = np.asfortranarray(np.array([
        [1.0, 2.0],
        [0.0, 1.0],
        [3.0, -1.0],
    ]))
    X = np.asfortranarray(np.zeros(B.shape, dtype=np.complex128))

    solver.solve_into(B, X)

    npt.assert_allclose(A @ X, B, atol=1e-12)


def test_complex_solve_into_requires_complex128_output():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)

    with pytest.raises(ValueError, match="complex128"):
        solver.solve_into(np.ones(3), np.zeros(3))


def test_complex_refactor_preserves_imaginary_values():
    A = _COMPLEX_CASES[4].values[1]
    matrix = _as_pardiso_csr(A, upper_only=False)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_NONSYM)

    A2 = A.copy()
    A2[np.diag_indices_from(A2)] += np.array([0.5j, -0.75j, 1.25j])
    matrix2 = _as_pardiso_csr(A2, upper_only=False)
    solver.refactor(matrix2.data.astype(np.complex128))

    b = np.array([1.0 + 2.0j, -1.0 + 0.5j, 3.0 - 1.0j])
    x = solver.solve(b)
    npt.assert_allclose(A2 @ x, b, atol=1e-12)


def test_complex64_refactor_values_are_promoted():
    A = _COMPLEX_CASES[4].values[1]
    matrix = _as_pardiso_csr(A, upper_only=False)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_NONSYM)

    A2 = A.copy()
    A2[np.diag_indices_from(A2)] += np.array([0.5j, -0.75j, 1.25j])
    matrix2 = _as_pardiso_csr(A2, upper_only=False)
    solver.refactor(matrix2.data.astype(np.complex64))

    b = np.array([1.0 + 2.0j, -1.0 + 0.5j, 3.0 - 1.0j])
    x = solver.solve(b)
    assert x.dtype == np.complex128
    npt.assert_allclose(A2 @ x, b, atol=1e-6)


def test_complex_run_phase_into():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = low_level.PardisoSolver(MTYPE_COMPLEX_HERM_POSDEF)
    _set_pattern(solver, matrix)
    solver.set_values(matrix.data.astype(np.complex128))
    solver.run_phase(11)
    b = np.array([1.0 + 0.5j, 2.0 - 1.0j, -1.0 + 2.0j])
    x = np.zeros(3, dtype=np.complex128)

    solver.run_phase_into(23, b, x)

    npt.assert_allclose(A @ x, b, atol=1e-12)


def test_real_high_level_solver_rejects_complex_matrix_values():
    matrix = sp.csr_matrix(np.array([
        [4.0 + 1.0j, 1.0],
        [0.0, 3.0],
    ]))

    with pytest.raises(ValueError, match="mtype is real"):
        PardisoSolver(matrix, MTYPE_REAL_NONSYM)


def test_real_low_level_solver_rejects_complex_values_and_rhs():
    A = sp.csr_matrix(np.array([[4.0, 1.0], [2.0, 3.0]]))
    solver = low_level.PardisoSolver(MTYPE_REAL_NONSYM)
    _set_pattern(solver, A)

    with pytest.raises(ValueError, match="mtype is real"):
        solver.set_values(A.data.astype(np.complex128))

    solver.factor(A.data.astype(np.float64))
    with pytest.raises(ValueError, match="mtype is real"):
        solver.solve(np.array([1.0 + 1.0j, 2.0]))


def test_complex_hermitian_type_rejects_lower_triangle():
    A = _COMPLEX_CASES[1].values[1]
    matrix = sp.csr_matrix(A)

    with pytest.raises(ValueError, match="upper-triangular"):
        PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)


@pytest.mark.parametrize(
    "mtype",
    [MTYPE_COMPLEX_HERM_POSDEF, MTYPE_COMPLEX_HERM_INDEF],
)
def test_hermitian_types_reject_imaginary_diagonal(mtype):
    A = np.array([
        [4.0 + 0.1j, 1.0 + 0.5j],
        [1.0 - 0.5j, 3.0],
    ])
    matrix = _as_pardiso_csr(A, upper_only=True)

    with pytest.raises(ValueError, match="diagonal entries must be real"):
        PardisoSolver(matrix, mtype)


def test_hermitian_refactor_rejects_imaginary_diagonal():
    A = _COMPLEX_CASES[1].values[1]
    matrix = _as_pardiso_csr(A, upper_only=True)
    solver = PardisoSolver(matrix, MTYPE_COMPLEX_HERM_POSDEF)
    invalid_values = matrix.data.copy()
    invalid_values[0] += 0.1j

    with pytest.raises(ValueError, match="row 0"):
        solver.refactor(invalid_values)


def test_invalid_mtype_errors_include_descriptions():
    with pytest.raises(ValueError, match="complex Hermitian positive definite"):
        low_level.PardisoSolver(999)

    with pytest.raises(ValueError, match="complex Hermitian positive definite"):
        PardisoSolver(sp.eye(2, format="csr"), 999)


def test_complex_matrix_type_constants():
    assert low_level.MTYPE_COMPLEX_STRUCT_SYM == 3
    assert low_level.MTYPE_COMPLEX_HERM_POSDEF == 4
    assert low_level.MTYPE_COMPLEX_HERM_INDEF == -4
    assert low_level.MTYPE_COMPLEX_SYM == 6
    assert low_level.MTYPE_COMPLEX_NONSYM == 13


def test_single_precision_iparm_is_rejected():
    solver = low_level.PardisoSolver(MTYPE_COMPLEX_NONSYM)
    with pytest.raises(ValueError, match=r"iparm\[27\]"):
        solver.set_iparm(27, 1)

    iparm = np.zeros(64, dtype=np.int64)
    iparm[0] = 1
    iparm[27] = 1
    iparm[34] = 1
    with pytest.raises(ValueError, match=r"iparm\[27\]"):
        solver.set_iparm_all(iparm)
