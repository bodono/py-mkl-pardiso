# py-mkl-pardiso

[![Test](https://github.com/bodono/py-mkl-pardiso/actions/workflows/test.yml/badge.svg)](https://github.com/bodono/py-mkl-pardiso/actions/workflows/test.yml)

Python pybind11 wrapper for the Intel oneMKL PARDISO sparse direct solver.

`pymklpardiso` exposes the real-valued subset of Intel's
[PARDISO](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/onemkl-pardiso-parallel-direct-sparse-solver-iface.html)
sparse direct solver to Python via [pybind11](https://github.com/pybind/pybind11).
It works with SciPy sparse matrices in CSR format and NumPy arrays.

**Supported platforms:** Linux (x86_64), Windows (AMD64).

## Installation

```bash
pip install py-mkl-pardiso
```

Or install from source (requires MKL):

```bash
git clone https://github.com/bodono/py-mkl-pardiso.git
cd py-mkl-pardiso
pip install -e ".[test]"
```

Building from source requires:
- Intel oneMKL (set `MKLROOT` if not auto-detected)
- A C++17 compiler
- Python >= 3.10
- pybind11 >= 2.12

## Quick start

```python
import numpy as np
import scipy.sparse as sp
from pymklpardiso import PardisoSolver, MTYPE_REAL_SYM_POSDEF

# Build a symmetric positive-definite matrix (upper triangle, CSR)
A_full = np.array([
    [4.0, 1.0],
    [1.0, 3.0],
])
A_upper = sp.csr_matrix(np.triu(A_full))
A_upper.sort_indices()

# Create solver, set pattern, factor, solve
solver = PardisoSolver(MTYPE_REAL_SYM_POSDEF)
solver.set_pattern(
    ia=A_upper.indptr.astype(np.int64),
    ja=A_upper.indices.astype(np.int64),
    n=A_upper.shape[0],
)
solver.factor(A_upper.data.astype(np.float64))

b = np.array([1.0, 2.0])
x = solver.solve(b)
print(x)  # [0.09090909 0.63636364]
```

## API reference

### `PardisoSolver(mtype, msglvl=0)`

Create a PARDISO solver instance.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mtype` | `int` | *(required)* | Matrix type (see constants below). |
| `msglvl` | `int` | `0` | Message level (0 = silent, 1 = print statistics). |

### Matrix type constants

| Constant | Value | Description |
|---|---|---|
| `MTYPE_REAL_STRUCT_SYM` | 1 | Real structurally symmetric |
| `MTYPE_REAL_SYM_POSDEF` | 2 | Real symmetric positive definite |
| `MTYPE_REAL_SYM_INDEF` | -2 | Real symmetric indefinite |
| `MTYPE_REAL_NONSYM` | 11 | Real nonsymmetric |

### Core methods

**`solver.set_pattern(ia, ja, n, check_sorted=True)`**
Set the CSR sparsity pattern. Uses zero-based indexing. Column indices must
be sorted within each row (unless `check_sorted=False`). For symmetric
positive definite and symmetric indefinite types, pass only the upper
triangle. For structurally symmetric and nonsymmetric types, pass the full
matrix.

**`solver.factor(a)`**
Set the nonzero values of the CSR matrix (i.e., `A_csr.data`) and factorize.
`a` must be a 1D array of length `nnz` matching the sparsity pattern from
`set_pattern()`. Runs symbolic analysis automatically if needed.

**`solver.solve(b)`**
Solve `Ax = b`. Accepts 1D `(n,)` or 2D `(n, nrhs)` arrays. Returns the
solution as a new NumPy array (Fortran-contiguous for 2D).

**`solver.solve_into(b, x)`**
Solve `Ax = b` writing into pre-allocated `x`. For 2D arrays, both `b` and
`x` must be Fortran-contiguous.

### Refactoring workflow

**`solver.set_values(a)`**
Load new nonzero values (i.e., `A_csr.data`) for the same sparsity pattern.
`a` must be a 1D array of length `nnz`.

**`solver.refactor()`**
Re-factorize using the currently loaded values.

```python
for new_values in value_generator:
    solver.set_values(new_values)
    solver.refactor()
    x = solver.solve(b)
```

### Other methods

| Method | Description |
|---|---|
| `solver.analyze()` | Run symbolic analysis (phase 11) explicitly. |
| `solver.release()` | Free PARDISO internal memory. |
| `solver.reset()` | Release and clear all state. |
| `solver.n()` | Matrix dimension. |
| `solver.nnz()` | Number of nonzeros in the sparsity pattern. |
| `solver.mtype()` | Matrix type. |
| `solver.set_perm(perm)` | Set fill-reducing permutation. |
| `solver.clear_perm()` | Clear permutation. |
| `solver.has_perm()` | Whether a permutation is set. |
| `solver.set_iparm(idx, value)` | Set a single iparm entry. |
| `solver.get_iparm()` | Get all 64 iparm values. |
| `solver.get_iparm_value(idx)` | Get a single iparm value. |
| `solver.set_iparm_all(iparm)` | Set all 64 iparm values. |
| `solver.set_msglvl(msglvl)` | Change message level. |
| `solver.run_phase(phase)` | Run an arbitrary PARDISO phase. |
| `solver.run_phase_into(phase, b, x)` | Run a phase with RHS/output arrays. |

### iparm notes

- `iparm[0]` is locked to `1` (user-supplied parameters).
- `iparm[34]` is locked to `1` (zero-based indexing).
- See the [MKL PARDISO iparm documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/pardiso-iparm-parameter.html) for all parameters.

## License

MIT
