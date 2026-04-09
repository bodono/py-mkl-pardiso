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

# Create solver — analyzes, factors, and is ready to solve
solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)

b = np.array([1.0, 2.0])
x = solver.solve(b)
print(x)  # [0.09090909 0.63636364]
```

### Refactoring workflow

When the sparsity pattern stays the same but values change (e.g., in an
iterative algorithm), use `refactor()` to skip symbolic analysis:

```python
solver = PardisoSolver(A_upper, MTYPE_REAL_SYM_POSDEF)
for new_values in value_generator:
    solver.refactor(new_values)
    x = solver.solve(b)
```

## API reference

### `PardisoSolver(A, mtype, iparms=None, msglvl=0)`

Create a PARDISO solver instance.  The constructor extracts the CSR sparsity
pattern from `A`, applies any `iparms` overrides, and runs symbolic analysis
+ numeric factorization so the solver is ready to call `solve()`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `A` | sparse CSR | *(required)* | Sparse matrix (any object with `indptr`, `indices`, `data`, `shape`). For symmetric types, pass only the upper triangle. |
| `mtype` | `int` | *(required)* | Matrix type (see constants below). |
| `iparms` | `dict` | `None` | Optional `{index: value}` iparm overrides. |
| `msglvl` | `int` | `0` | Message level (0 = silent, 1 = print statistics). |

### Matrix type constants

| Constant | Value | Description |
|---|---|---|
| `MTYPE_REAL_STRUCT_SYM` | 1 | Real structurally symmetric |
| `MTYPE_REAL_SYM_POSDEF` | 2 | Real symmetric positive definite |
| `MTYPE_REAL_SYM_INDEF` | -2 | Real symmetric indefinite |
| `MTYPE_REAL_NONSYM` | 11 | Real nonsymmetric |

### Core methods

**`solver.solve(b)`**
Solve `Ax = b`. Accepts 1D `(n,)` or 2D `(n, nrhs)` arrays. Returns the
solution as a new NumPy array (Fortran-contiguous for 2D).

**`solver.solve_into(b, x)`**
Solve `Ax = b` writing into pre-allocated `x`. For 2D arrays, both `b` and
`x` must be Fortran-contiguous.

**`solver.refactor(values)`**
Re-factorize with new nonzero values (phase 22 only). Does not re-run
symbolic analysis. Use `factor()` to re-analyze from scratch.

**`solver.factor(values)`**
Re-analyze and re-factorize with new values (phases 11 + 22). Use this for
error recovery or when iparm changes require fresh symbolic analysis.

### Other methods

| Method | Description |
|---|---|
| `solver.release()` | Free PARDISO internal memory. |
| `solver.n` | Matrix dimension (property). |
| `solver.nnz` | Number of nonzeros (property). |
| `solver.mtype` | Matrix type (property). |
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
