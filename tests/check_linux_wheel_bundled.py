import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[(.+)\]")
MKL_LIBS = (
    "libmkl_intel_ilp64",
    "libmkl_sequential",
    "libmkl_core",
)


def find_extension(root):
    sos = list(root.rglob("*.so"))
    if not sos:
        raise RuntimeError("No .so file found in extracted wheel")
    if len(sos) > 1:
        raise RuntimeError(
            "Expected exactly one .so file in wheel, found:\n"
            + "\n".join(str(path) for path in sos)
        )
    return sos[0]


def imported_shared_libs(path):
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    imports = []
    for line in result.stdout.splitlines():
        match = NEEDED_RE.search(line)
        if match:
            imports.append(match.group(1).lower())
    return sorted(imports)


def has_pardiso_symbols(path):
    """Check that PARDISO symbols are present (statically linked)."""
    result = subprocess.run(
        ["nm", "-g", str(path)],
        capture_output=True,
        text=True,
    )
    return "pardiso_64" in result.stdout.lower() or "pardiso" in result.stdout.lower()


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
            wheel_files = [Path(name).name.lower() for name in wheel.namelist()]
        extension = find_extension(Path(tmp))
        imports = imported_shared_libs(extension)

    # With static linking, the extension should NOT depend on MKL shared libs.
    dynamic_mkl = [
        imp for imp in imports
        if any(imp.startswith(lib) for lib in MKL_LIBS)
    ]
    if dynamic_mkl:
        raise RuntimeError(
            f"{path} dynamically links MKL (expected static): {dynamic_mkl}"
        )

    # No separate MKL .so files should be bundled in the wheel.
    bundled_mkl = [
        name for name in wheel_files
        if any(name.startswith(lib) and ".so" in name for lib in MKL_LIBS)
    ]
    if bundled_mkl:
        raise RuntimeError(
            f"{path} bundles MKL shared libraries (expected static): {bundled_mkl}"
        )

    # Verify PARDISO symbols are present (MKL is statically linked).
    with tempfile.TemporaryDirectory() as tmp2:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp2)
        ext = find_extension(Path(tmp2))
        if not has_pardiso_symbols(ext):
            raise RuntimeError(
                f"{path} does not contain PARDISO symbols — MKL may not be linked"
            )

    print(f"{path}: MKL is statically linked (PARDISO symbols present, no dynamic MKL deps)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Linux wheels have MKL statically linked."
    )
    parser.add_argument("wheelhouse", help="Directory containing built wheel files")
    args = parser.parse_args()

    wheelhouse = Path(args.wheelhouse)
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheels found in {wheelhouse}")

    for wheel in wheels:
        check_wheel(wheel)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
