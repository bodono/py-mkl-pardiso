// py-mkl-pardiso.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mkl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using Index = MKL_INT64;

[[noreturn]] void throw_value_error(const std::string& msg) {
    throw py::value_error(msg);
}

[[noreturn]] void throw_runtime_error(const std::string& msg) {
    throw std::runtime_error(msg);
}

std::string pardiso_error_string(Index err) {
    switch (err) {
        case 0:   return "success";
        case -1:  return "input inconsistent";
        case -2:  return "not enough memory";
        case -3:  return "reordering problem";
        case -4:  return "zero pivot / numerical factorization failure";
        case -5:  return "unclassified internal error";
        case -6:  return "reordering failed";
        case -7:  return "diagonal matrix problem";
        case -8:  return "32-bit integer overflow";
        case -9:  return "not enough out-of-core memory";
        case -10: return "error opening out-of-core files";
        case -11: return "read/write out-of-core file error";
        default: {
            std::ostringstream oss;
            oss << "unknown PARDISO error " << err;
            return oss.str();
        }
    }
}

template <typename T, int Extra>
void require_1d(const py::array_t<T, Extra>& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw_value_error(std::string(name) + " must be a 1D NumPy array");
    }
}

void check_sorted_rows(const std::vector<Index>& ia, const std::vector<Index>& ja) {
    const Index n = static_cast<Index>(ia.size()) - 1;
    for (Index i = 0; i < n; ++i) {
        const std::size_t start = static_cast<std::size_t>(ia[static_cast<std::size_t>(i)]);
        const std::size_t end = static_cast<std::size_t>(ia[static_cast<std::size_t>(i + 1)]);
        if (!std::is_sorted(ja.begin() + start, ja.begin() + end)) {
            std::ostringstream oss;
            oss << "CSR column indices must be sorted within each row; row "
                << i << " is not sorted";
            throw_value_error(oss.str());
        }
    }
}

}  // namespace

class PardisoSolver {
public:
    explicit PardisoSolver(Index mtype, Index msglvl = 0)
        : mtype_(mtype),
          maxfct_(1),
          mnum_(1),
          msglvl_(msglvl),
          n_(0),
          pattern_set_(false),
          values_set_(false),
          analyzed_(false),
          factored_(false) {
        if (mtype != 1 && mtype != 2 && mtype != -2 && mtype != 11) {
            throw_value_error(
                "mtype must be one of: 1 (structurally symmetric), "
                "2 (symmetric positive definite), -2 (symmetric indefinite), "
                "11 (nonsymmetric)");
        }
        init_pardiso_state();
    }

    ~PardisoSolver() {
        try {
            release();
        } catch (...) {
        }
    }

    void reset() {
        release();
        clear_pattern_values_perm();
        init_pardiso_state();
    }

    Index n() const { return n_; }
    Index nnz() const { return static_cast<Index>(a_.size()); }
    Index mtype() const { return mtype_; }

    std::vector<Index> get_iparm() const {
        return std::vector<Index>(iparm_.begin(), iparm_.end());
    }

    Index get_iparm_value(int idx) const {
        check_iparm_index(idx);
        return iparm_[static_cast<std::size_t>(idx)];
    }

    void set_iparm(int idx, Index value) {
        check_iparm_index(idx);

        // These are wrapper invariants.
        if (idx == 0) {
            if (value != 1) {
                throw_value_error("iparm[0] must remain 1 in this wrapper");
            }
            iparm_[0] = 1;
            return;
        }
        if (idx == 34) {
            if (value != 1) {
                throw_value_error("iparm[34] must remain 1 because this wrapper always uses zero-based indexing");
            }
            iparm_[34] = 1;
            return;
        }

        maybe_invalidate_handle_for_iparm_change(idx, value);
        iparm_[static_cast<std::size_t>(idx)] = value;
    }

    void set_iparm_all(py::array_t<Index, py::array::c_style | py::array::forcecast> iparm) {
        require_1d(iparm, "iparm");
        if (iparm.size() != 64) {
            throw_value_error("iparm must have length 64");
        }

        auto u = iparm.unchecked<1>();

        if (u(0) != 1) {
            throw_value_error("iparm[0] must be 1 in this wrapper");
        }
        if (u(34) != 1) {
            throw_value_error("iparm[34] must be 1 because this wrapper always uses zero-based indexing");
        }

        // Invalidate first if a phase-1-sensitive parameter changes.
        for (int i = 0; i < 64; ++i) {
            maybe_invalidate_handle_for_iparm_change(i, u(i));
        }

        std::copy(u.data(0), u.data(0) + 64, iparm_.begin());

        // Reassert wrapper-owned invariants.
        iparm_[0] = 1;
        iparm_[34] = 1;
    }

    void set_msglvl(Index msglvl) {
        msglvl_ = msglvl;
    }

    void set_pattern(
        py::array_t<Index, py::array::c_style | py::array::forcecast> ia,
        py::array_t<Index, py::array::c_style | py::array::forcecast> ja,
        Index n,
        bool check_sorted = true
    ) {
        require_1d(ia, "ia");
        require_1d(ja, "ja");

        if (n <= 0) {
            throw_value_error("n must be positive");
        }
        if (ia.size() != static_cast<py::ssize_t>(n + 1)) {
            throw_value_error("ia must have length n + 1");
        }

        auto ia_u = ia.unchecked<1>();
        auto ja_u = ja.unchecked<1>();

        const Index nnz = ia_u(n);
        if (nnz < 0) {
            throw_value_error("ia[n] must be nonnegative");
        }
        if (ja.size() != static_cast<py::ssize_t>(nnz)) {
            throw_value_error("ja length must equal ia[n]");
        }

        for (Index i = 0; i < n; ++i) {
            if (ia_u(i) > ia_u(i + 1)) {
                throw_value_error("ia must be nondecreasing");
            }
        }
        for (Index k = 0; k < nnz; ++k) {
            if (ja_u(k) < 0 || ja_u(k) >= n) {
                throw_value_error("ja contains out-of-range column index");
            }
        }

        release();
        clear_pattern_values_perm();

        n_ = n;
        ia_.assign(ia_u.data(0), ia_u.data(0) + (n + 1));
        ja_.assign(ja_u.data(0), ja_u.data(0) + nnz);
        a_.assign(static_cast<std::size_t>(nnz), 0.0);

        if (check_sorted) {
            check_sorted_rows(ia_, ja_);
        }

        pattern_set_ = true;
        values_set_ = false;
        analyzed_ = false;
        factored_ = false;
    }

    void set_values(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        require_1d(a, "a");
        ensure_pattern();

        if (a.size() != static_cast<py::ssize_t>(a_.size())) {
            throw_value_error("a length must match the current sparsity pattern nnz");
        }

        auto a_u = a.unchecked<1>();
        std::copy(a_u.data(0), a_u.data(0) + a_.size(), a_.begin());

        values_set_ = true;
        factored_ = false;

        // If phase 1 depends on values, changing values invalidates previous analysis.
        if (analysis_depends_on_values()) {
            analyzed_ = false;
        }
    }

    void set_perm(py::array_t<Index, py::array::c_style | py::array::forcecast> perm) {
        require_1d(perm, "perm");
        ensure_pattern();

        if (perm.size() != static_cast<py::ssize_t>(n_)) {
            throw_value_error("perm must have length n");
        }

        auto p_u = perm.unchecked<1>();
        perm_.assign(p_u.data(0), p_u.data(0) + n_);
    }

    void clear_perm() {
        perm_.clear();
    }

    bool has_perm() const {
        return !perm_.empty();
    }

    void analyze() {
        ensure_pattern();
        validate_common_preconditions(/*phase=*/11);
        call_pardiso(/*phase=*/11, /*nrhs=*/1, nullptr, nullptr);
        analyzed_ = true;
        factored_ = false;
    }

    void factor(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        set_values(a);
        factor_loaded_values();
    }

    void refactor() {
        ensure_pattern();
        ensure_values();
        if (!analyzed_) {
            call_pardiso(/*phase=*/11, /*nrhs=*/1, nullptr, nullptr);
            analyzed_ = true;
        }
        call_pardiso(/*phase=*/22, /*nrhs=*/1, nullptr, nullptr);
        factored_ = true;
    }

    void refactor_values(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
        set_values(a);
        refactor();
    }

    py::array_t<double> solve(py::array_t<double, py::array::forcecast> b) {
        ensure_factored();

        if (b.ndim() == 1) {
            if (b.shape(0) != n_) {
                throw_value_error("1D rhs must have length n");
            }

            auto b_c = py::array_t<double, py::array::c_style | py::array::forcecast>(b);
            py::array_t<double> x({static_cast<py::ssize_t>(n_)});
            auto bbuf = b_c.request();
            auto xbuf = x.request();

            call_pardiso(/*phase=*/33, /*nrhs=*/1,
                         static_cast<double*>(bbuf.ptr),
                         static_cast<double*>(xbuf.ptr));
            return x;
        }

        if (b.ndim() == 2) {
            if (b.shape(0) != n_) {
                throw_value_error("2D rhs must have shape (n, nrhs)");
            }

            const Index nrhs = static_cast<Index>(b.shape(1));

            // PARDISO expects column-major (Fortran) layout for multi-RHS.
            auto b_f = py::array_t<double, py::array::f_style | py::array::forcecast>(b);

            // Create Fortran-contiguous output array.
            const py::ssize_t n_s = static_cast<py::ssize_t>(n_);
            const py::ssize_t nrhs_s = static_cast<py::ssize_t>(nrhs);
            const py::ssize_t dsize = static_cast<py::ssize_t>(sizeof(double));
            py::array_t<double> x(
                std::vector<py::ssize_t>{n_s, nrhs_s},
                std::vector<py::ssize_t>{dsize, n_s * dsize}
            );

            auto bbuf = b_f.request();
            auto xbuf = x.request();

            call_pardiso(/*phase=*/33, nrhs,
                         static_cast<double*>(bbuf.ptr),
                         static_cast<double*>(xbuf.ptr));
            return x;
        }

        throw_value_error("rhs must be 1D or 2D");
    }

    void solve_into(
        py::array_t<double, py::array::forcecast> b,
        py::array_t<double, py::array::forcecast> x
    ) {
        ensure_factored();

        const Index nrhs = validate_rhs_pair(b, x);

        // For multi-RHS, PARDISO expects column-major layout.
        if (b.ndim() == 2) {
            ensure_f_contiguous(b, "b");
            ensure_f_contiguous(x, "x");
        } else {
            ensure_contiguous(b, "b");
            ensure_contiguous(x, "x");
        }

        auto bbuf = b.request();
        auto xbuf = x.request();

        call_pardiso(/*phase=*/33, nrhs,
                     static_cast<double*>(bbuf.ptr),
                     static_cast<double*>(xbuf.ptr));
    }

    void run_phase(Index phase) {
        if (phase == -1) {
            release();
            return;
        }

        ensure_pattern();
        validate_common_preconditions(phase);
        call_pardiso(phase, /*nrhs=*/1, nullptr, nullptr);
        update_state_after_phase(phase);
    }

    void run_phase_into(
        Index phase,
        py::array_t<double, py::array::forcecast> b,
        py::array_t<double, py::array::forcecast> x
    ) {
        if (phase == -1) {
            throw_value_error("use release() for phase -1");
        }

        ensure_pattern();
        const Index nrhs = validate_rhs_pair(b, x);
        validate_common_preconditions(phase);

        // For multi-RHS, PARDISO expects column-major layout.
        if (b.ndim() == 2) {
            ensure_f_contiguous(b, "b");
            ensure_f_contiguous(x, "x");
        } else {
            ensure_contiguous(b, "b");
            ensure_contiguous(x, "x");
        }

        auto bbuf = b.request();
        auto xbuf = x.request();

        call_pardiso(phase, nrhs,
                     static_cast<double*>(bbuf.ptr),
                     static_cast<double*>(xbuf.ptr));
        update_state_after_phase(phase);
    }

    void release() {
        if (!owns_handle()) {
            analyzed_ = false;
            factored_ = false;
            return;
        }

        Index phase = -1;
        Index nrhs = 1;
        Index idum = 0;
        double ddum = 0.0;
        Index error = 0;

        Index* p_ptr = ensure_perm_buffer();

        {
            py::gil_scoped_release nogil;
            pardiso_64(
                pt_.data(),
                &maxfct_,
                &mnum_,
                &mtype_,
                &phase,
                &n_,
                &ddum,
                ia_.empty() ? &idum : ia_.data(),
                ja_.empty() ? &idum : ja_.data(),
                p_ptr,
                &nrhs,
                iparm_.data(),
                &msglvl_,
                &ddum,
                &ddum,
                &error
            );
        }

        std::fill(pt_.begin(), pt_.end(), nullptr);
        analyzed_ = false;
        factored_ = false;

        if (error != 0) {
            throw_runtime_error("PARDISO release failed: " + pardiso_error_string(error));
        }
    }

private:
    void init_pardiso_state() {
        std::fill(pt_.begin(), pt_.end(), nullptr);
        std::fill(iparm_.begin(), iparm_.end(), 0);

        // Wrapper-owned invariants.
        iparm_[0] = 1;   // use user-supplied iparm
        iparm_[34] = 1;  // zero-based indexing
    }

    void clear_pattern_values_perm() {
        ia_.clear();
        ja_.clear();
        a_.clear();
        perm_.clear();
        perm_buf_.clear();
        n_ = 0;
        pattern_set_ = false;
        values_set_ = false;
    }

    void check_iparm_index(int idx) const {
        if (idx < 0 || idx >= 64) {
            throw_value_error("iparm index must be in [0, 63]");
        }
    }

    bool owns_handle() const {
        for (void* p : pt_) {
            if (p != nullptr) {
                return true;
            }
        }
        return false;
    }

    void ensure_pattern() const {
        if (!pattern_set_) {
            throw_runtime_error("set_pattern() must be called first");
        }
    }

    void ensure_values() const {
        if (!values_set_) {
            throw_runtime_error("numeric values are not set; call set_values(a) or factor(a)");
        }
    }

    void ensure_factored() const {
        if (!factored_) {
            throw_runtime_error("matrix is not factored; call factor(a) first");
        }
    }

    Index* ensure_perm_buffer() {
        if (!perm_.empty()) {
            return perm_.data();
        }
        if (n_ > 0) {
            perm_buf_.resize(static_cast<std::size_t>(n_), 0);
        }
        return perm_buf_.data();
    }

    static void ensure_contiguous(const py::array& arr, const char* name) {
        if ((arr.flags() & (py::array::c_style | py::array::f_style)) == 0) {
            throw_value_error(std::string(name) + " must be contiguous");
        }
    }

    static void ensure_f_contiguous(const py::array& arr, const char* name) {
        if (!(arr.flags() & py::array::f_style)) {
            throw_value_error(std::string(name) + " must be Fortran-contiguous (column-major) for multi-RHS solve");
        }
    }

    bool analysis_depends_on_values() const {
        // Common value-dependent analysis options in MKL PARDISO.
        return iparm_[10] != 0 || iparm_[12] != 0;
    }

    void maybe_invalidate_handle_for_iparm_change(int idx, Index new_value) {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (iparm_[i] == new_value) {
            return;
        }

        // Many iparm entries affect symbolic analysis (e.g., iparm[1]
        // reordering, iparm[3] preconditioned CGS, iparm[9] pivoting,
        // iparm[27] single/double precision, iparm[33] CNR mode, etc.).
        // Conservatively release the handle on any change to avoid stale
        // analysis results.
        if (analyzed_ && owns_handle()) {
            release();
        }
    }

    void validate_common_preconditions(Index phase) const {
        if ((phase == 12 || phase == 22 || phase == 23) && !values_set_) {
            throw_runtime_error("this phase requires numeric values; call set_values(a) first");
        }

        if ((phase == 33 || phase == 331 || phase == 332 || phase == 333) && !factored_) {
            throw_runtime_error("this phase requires prior factorization");
        }

        if ((iparm_[30] != 0 || iparm_[35] != 0) && perm_.empty()) {
            throw_runtime_error("perm must be set when iparm[30] or iparm[35] is enabled");
        }
    }

    void factor_loaded_values() {
        ensure_values();

        if (!analyzed_ || analysis_depends_on_values()) {
            call_pardiso(/*phase=*/11, /*nrhs=*/1, nullptr, nullptr);
            analyzed_ = true;
        }

        call_pardiso(/*phase=*/22, /*nrhs=*/1, nullptr, nullptr);
        factored_ = true;
    }

    Index validate_rhs_pair(
        const py::array_t<double, py::array::forcecast>& b,
        const py::array_t<double, py::array::forcecast>& x
    ) const {
        if (b.ndim() != x.ndim()) {
            throw_value_error("b and x must have the same rank");
        }

        if (b.ndim() == 1) {
            if (b.shape(0) != n_ || x.shape(0) != n_) {
                throw_value_error("1D b and x must each have length n");
            }
            return 1;
        }

        if (b.ndim() == 2) {
            if (b.shape(0) != n_ || x.shape(0) != n_) {
                throw_value_error("2D b and x must have shape (n, nrhs)");
            }
            if (b.shape(1) != x.shape(1)) {
                throw_value_error("b and x must have the same number of right-hand sides");
            }
            return static_cast<Index>(b.shape(1));
        }

        throw_value_error("b and x must be 1D or 2D");
    }

    void update_state_after_phase(Index phase) {
        switch (phase) {
            case 11:
                analyzed_ = true;
                factored_ = false;
                break;
            case 12:
            case 22:
            case 23:
                analyzed_ = true;
                factored_ = true;
                break;
            default:
                break;
        }
    }

    void call_pardiso(Index phase, Index nrhs, double* b, double* x) {
        ensure_pattern();

        Index idum = 0;
        double ddum = 0.0;
        Index error = 0;

        double* a_ptr = a_.empty() ? &ddum : a_.data();
        Index* ia_ptr = ia_.empty() ? &idum : ia_.data();
        Index* ja_ptr = ja_.empty() ? &idum : ja_.data();

        // PARDISO writes the fill-in reducing permutation (size n) to perm
        // during phase 11, so we must always provide a properly-sized buffer.
        Index* p_ptr = ensure_perm_buffer();

        {
            py::gil_scoped_release nogil;
            pardiso_64(
                pt_.data(),
                &maxfct_,
                &mnum_,
                &mtype_,
                &phase,
                &n_,
                a_ptr,
                ia_ptr,
                ja_ptr,
                p_ptr,
                &nrhs,
                iparm_.data(),
                &msglvl_,
                b ? b : &ddum,
                x ? x : &ddum,
                &error
            );
        }

        if (error != 0) {
            std::ostringstream oss;
            oss << "PARDISO phase " << phase << " failed: "
                << pardiso_error_string(error);
            throw_runtime_error(oss.str());
        }
    }

    std::array<void*, 64> pt_{};
    std::array<Index, 64> iparm_{};

    Index mtype_;
    Index maxfct_;
    Index mnum_;
    Index msglvl_;
    Index n_;

    bool pattern_set_;
    bool values_set_;
    bool analyzed_;
    bool factored_;

    std::vector<Index> ia_;
    std::vector<Index> ja_;
    std::vector<double> a_;
    std::vector<Index> perm_;
    std::vector<Index> perm_buf_;  // Reusable buffer for PARDISO perm when user hasn't set one.
};

PYBIND11_MODULE(_mkl_pardiso, m) {
    m.doc() = "pybind11 wrapper for common Intel oneMKL PARDISO real-valued use cases";

    py::class_<PardisoSolver>(m, "PardisoSolver")
        .def(py::init<Index, Index>(),
             py::arg("mtype"),
             py::arg("msglvl") = 0)

        .def("reset", &PardisoSolver::reset)

        .def("n", &PardisoSolver::n)
        .def("nnz", &PardisoSolver::nnz)
        .def("mtype", &PardisoSolver::mtype)

        .def("get_iparm", &PardisoSolver::get_iparm)
        .def("get_iparm_value", &PardisoSolver::get_iparm_value, py::arg("idx"))
        .def("set_iparm", &PardisoSolver::set_iparm, py::arg("idx"), py::arg("value"))
        .def("set_iparm_all", &PardisoSolver::set_iparm_all, py::arg("iparm"))
        .def("set_msglvl", &PardisoSolver::set_msglvl, py::arg("msglvl"))

        .def("set_pattern", &PardisoSolver::set_pattern,
             py::arg("ia"), py::arg("ja"), py::arg("n"),
             py::arg("check_sorted") = true,
             R"doc(
Set the CSR sparsity pattern.

Notes:
- Uses zero-based indexing.
- CSR column indices should be sorted within each row.
- For symmetric matrix types, pass only the required triangle, not both.
)doc")

        .def("set_values", &PardisoSolver::set_values, py::arg("a"))
        .def("set_perm", &PardisoSolver::set_perm, py::arg("perm"))
        .def("clear_perm", &PardisoSolver::clear_perm)
        .def("has_perm", &PardisoSolver::has_perm)

        .def("analyze", &PardisoSolver::analyze)
        .def("factor", &PardisoSolver::factor, py::arg("a"),
             R"doc(
Set numeric values and perform numerical factorization.
If symbolic analysis has not been run yet, it is run automatically first.
)doc")
        .def("refactor", &PardisoSolver::refactor,
             R"doc(
Refactor using the currently loaded numeric values (phase 22 only).
Does not re-run symbolic analysis. Use factor(a) to re-analyze.
)doc")
        .def("refactor_values", &PardisoSolver::refactor_values, py::arg("a"),
             R"doc(
Set new numeric values and refactor (phase 22 only).
Equivalent to set_values(a) followed by refactor().
)doc")

        .def("solve", &PardisoSolver::solve, py::arg("b"))
        .def("solve_into", &PardisoSolver::solve_into, py::arg("b"), py::arg("x"))

        .def("run_phase", &PardisoSolver::run_phase, py::arg("phase"),
             R"doc(
Run a PARDISO phase that does not require explicit RHS/output arrays.
Phase -1 is accepted and delegates to release().
)doc")
        .def("run_phase_into", &PardisoSolver::run_phase_into,
             py::arg("phase"), py::arg("b"), py::arg("x"),
             R"doc(
Run a PARDISO phase with explicit RHS and output arrays.
For phase -1, use release().
)doc")

        .def("release", &PardisoSolver::release);

    // Common real-valued matrix types.
    m.attr("MTYPE_REAL_STRUCT_SYM") = py::int_(1);
    m.attr("MTYPE_REAL_SYM_INDEF")  = py::int_(-2);
    m.attr("MTYPE_REAL_SYM_POSDEF") = py::int_(2);
    m.attr("MTYPE_REAL_NONSYM")     = py::int_(11);
}
