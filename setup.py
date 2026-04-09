import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def find_mkl():
    """Find MKL include and library directories.

    Search order:
      1. MKLROOT environment variable
      2. CONDA_PREFIX environment variable
      3. Standard Intel oneAPI paths
      4. System-installed MKL (apt libmkl-dev on Ubuntu/Debian)
    """
    # Candidate base directories (MKLROOT-style layout: include/, lib/).
    candidates = []

    mklroot = os.environ.get("MKLROOT")
    if mklroot:
        candidates.append(mklroot)

    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        candidates.append(conda)

    if sys.platform == "linux":
        candidates.append("/opt/intel/oneapi/mkl/latest")
    elif sys.platform == "win32":
        pf = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates.append(os.path.join(pf, "Intel", "oneAPI", "mkl", "latest"))

    # On Windows, conda-forge packages install under a Library/ subdirectory.
    if sys.platform == "win32":
        candidates += [os.path.join(c, "Library") for c in candidates]

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

    # Ubuntu/Debian: apt-get install libmkl-dev
    if sys.platform == "linux":
        for inc in ("/usr/include/mkl", "/usr/include"):
            if os.path.isfile(os.path.join(inc, "mkl.h")):
                return inc, "/usr/lib/x86_64-linux-gnu"

    raise RuntimeError(
        "Could not find MKL. Set the MKLROOT environment variable to your "
        "MKL installation, or install via conda: conda install mkl-devel"
    )


# MKL is required at compile time but not for sdist creation.
try:
    mkl_include, mkl_libdir = find_mkl()
except RuntimeError:
    mkl_include = ""
    mkl_libdir = ""

# ILP64 interface (64-bit integers) -- matches MKL_INT64 / pardiso_64 in C++ source.
define_macros = [("MKL_ILP64", None)]

# When PYMKLPARDISO_STATIC=1, statically link MKL so the wheel is
# self-contained and users don't need MKL installed at runtime.
static = os.environ.get("PYMKLPARDISO_STATIC", "").lower() in ("1", "true", "yes")

if sys.platform == "win32":
    libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core"]
    extra_link_args = []
elif static:
    # Static linking on Linux: pass full paths to .a archives inside
    # --start-group / --end-group to resolve circular MKL dependencies.
    libraries = []
    _mkl_archives = [
        os.path.join(mkl_libdir, f"lib{name}.a")
        for name in ("mkl_intel_ilp64", "mkl_sequential", "mkl_core")
    ]
    extra_link_args = (
        ["-Wl,--start-group"] + _mkl_archives +
        ["-Wl,--end-group", "-lpthread", "-lm", "-ldl"]
    )
else:
    # Dynamic linking (for local development / testing).
    libraries = ["mkl_intel_ilp64", "mkl_sequential", "mkl_core", "pthread", "m", "dl"]
    extra_link_args = [f"-Wl,-rpath,{mkl_libdir}"]

ext_modules = [
    Pybind11Extension(
        "pymklpardiso._mkl_pardiso",
        ["py-mkl-pardiso.cpp"],
        cxx_std=17,
        include_dirs=[mkl_include],
        library_dirs=[mkl_libdir] if libraries else [],
        libraries=libraries,
        define_macros=define_macros,
        extra_link_args=extra_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
