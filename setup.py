import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _find_mkl_via_pip():
    """Try to locate MKL headers and libraries from pip-installed packages."""
    try:
        import mkl_include
        inc = mkl_include.get_include()
    except (ImportError, AttributeError):
        return None

    if not os.path.isfile(os.path.join(inc, "mkl.h")):
        return None

    # mkl-devel / mkl-static put libraries alongside the package.
    for pkg_name in ("mkl_devel", "mkl_static", "mkl"):
        try:
            mod = __import__(pkg_name)
            lib = os.path.join(os.path.dirname(mod.__file__), "lib")
            if os.path.isdir(lib):
                return inc, lib
        except ImportError:
            continue

    return None


def find_mkl():
    """Find MKL include and library directories."""
    candidates = []

    # 1. MKLROOT environment variable (Intel oneAPI / standalone MKL)
    mklroot = os.environ.get("MKLROOT")
    if mklroot:
        candidates.append(mklroot)

    # 2. CONDA_PREFIX (conda-installed MKL)
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(conda)

    # 3. Standard Intel oneAPI paths
    if sys.platform == "linux":
        candidates.append("/opt/intel/oneapi/mkl/latest")
    elif sys.platform == "win32":
        pf = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates.append(os.path.join(pf, "Intel", "oneAPI", "mkl", "latest"))

    for base in candidates:
        inc = os.path.join(base, "include")
        if sys.platform == "win32":
            lib = os.path.join(base, "lib")
        else:
            lib = os.path.join(base, "lib", "intel64")
            if not os.path.isdir(lib):
                lib = os.path.join(base, "lib")
        if os.path.isfile(os.path.join(inc, "mkl.h")):
            return inc, lib

    # 4. Fallback: pip-installed MKL packages (mkl-include + mkl-devel/mkl-static)
    pip_result = _find_mkl_via_pip()
    if pip_result is not None:
        return pip_result

    raise RuntimeError(
        "Could not find MKL. Set the MKLROOT environment variable to your "
        "MKL installation (e.g. /opt/intel/oneapi/mkl/latest), or "
        "pip install mkl-include mkl-devel."
    )


mkl_include, mkl_libdir = find_mkl()

# ILP64 interface (64-bit integers) -- matches MKL_INT64 / pardiso_64 in C++ source.
define_macros = [("MKL_ILP64", None)]

if sys.platform == "win32":
    libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core"]
    extra_link_args = []
else:
    libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core", "pthread", "m", "dl"]
    extra_link_args = [f"-Wl,-rpath,{mkl_libdir}"]

ext_modules = [
    Pybind11Extension(
        "pymklpardiso._mkl_pardiso",
        ["py-mkl-pardiso.cpp"],
        cxx_std=17,
        include_dirs=[mkl_include],
        library_dirs=[mkl_libdir],
        libraries=libraries,
        define_macros=define_macros,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
