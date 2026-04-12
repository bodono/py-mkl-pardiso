import argparse
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pefile


def find_extension(root):
    pyds = list(root.rglob("*.pyd"))
    if not pyds:
        raise RuntimeError("No .pyd file found in extracted wheel")
    if len(pyds) > 1:
        raise RuntimeError(
            "Expected exactly one .pyd file in wheel, found:\n"
            + "\n".join(str(path) for path in pyds)
        )
    return pyds[0]


def imported_dlls(path):
    pe = pefile.PE(str(path))
    try:
        return sorted(
            entry.dll.decode("ascii").lower()
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        )
    finally:
        pe.close()


def has_pardiso_strings(path):
    """Check that PARDISO-related strings are present (statically linked MKL)."""
    data = Path(path).read_bytes().lower()
    return b"pardiso" in data


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
            wheel_files = [Path(name).name.lower() for name in wheel.namelist()]
        extension = find_extension(Path(tmp))
        imports = imported_dlls(extension)

        # With static linking, the extension should NOT import MKL DLLs.
        mkl_imports = [
            name for name in imports
            if name.startswith("mkl") and name.endswith(".dll")
        ]
        if mkl_imports:
            raise RuntimeError(
                f"{path} dynamically imports MKL DLLs (expected static): {mkl_imports}"
            )

        # No separate MKL DLLs should be bundled in the wheel.
        bundled_mkl = [
            name for name in wheel_files
            if name.startswith("mkl") and name.endswith(".dll")
        ]
        if bundled_mkl:
            raise RuntimeError(
                f"{path} bundles MKL DLLs (expected static): {bundled_mkl}"
            )

        # Verify the extension imports from the Python DLL (valid pybind11 module).
        python_imports = [name for name in imports if name.startswith("python")]
        if not python_imports:
            raise RuntimeError(
                f"{path} does not import from Python DLL — may not be a valid extension"
            )

        # Verify PARDISO symbols are present (MKL is statically linked).
        if not has_pardiso_strings(extension):
            raise RuntimeError(
                f"{path} does not contain PARDISO strings — MKL may not be linked"
            )

    print(f"{path}: MKL is statically linked (PARDISO strings present, no dynamic MKL deps)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Windows wheels have MKL statically linked."
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
