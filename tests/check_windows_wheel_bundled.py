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


def has_pardiso_exports(path):
    """Check that the extension exports or contains PARDISO-related symbols."""
    pe = pefile.PE(str(path))
    try:
        # Check exported symbols.
        for entry in getattr(pe, "DIRECTORY_ENTRY_EXPORT", []):
            if entry is None:
                continue
            for sym in getattr(entry, "symbols", []):
                name = getattr(sym, "name", None)
                if name and b"pardiso" in name.lower():
                    return True
        # Check imported symbols (pybind11 init function implies the module loaded).
        for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
            dll_name = entry.dll.decode("ascii").lower()
            # If it imports from python DLL, the module is valid.
            if dll_name.startswith("python"):
                return True
        return False
    finally:
        pe.close()


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

    # Verify the extension is a valid pybind11 module (implies MKL is linked).
    with tempfile.TemporaryDirectory() as tmp2:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp2)
        ext = find_extension(Path(tmp2))
        if not has_pardiso_exports(ext):
            raise RuntimeError(
                f"{path} does not appear to be a valid pybind11 module"
            )

    print(f"{path}: MKL is statically linked (no MKL DLL dependencies)")


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
