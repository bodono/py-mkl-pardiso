import argparse
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pefile


EXPECTED_MKL_DLLS = (
    "mkl_intel_ilp64",
    "mkl_sequential",
    "mkl_core",
)


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


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
            wheel_files = [Path(name).name.lower() for name in wheel.namelist()]
        extension = find_extension(Path(tmp))
        imports = imported_dlls(extension)

    missing_imports = [
        lib for lib in EXPECTED_MKL_DLLS
        if not any(name.startswith(lib) and name.endswith(".dll") for name in imports)
    ]
    if missing_imports:
        raise RuntimeError(
            f"{path} is missing dynamic MKL DLL imports:\n" + "\n".join(missing_imports)
        )

    missing_bundles = [
        lib for lib in EXPECTED_MKL_DLLS
        if not any(name.startswith(lib) and name.endswith(".dll") for name in wheel_files)
    ]
    if missing_bundles:
        raise RuntimeError(
            f"{path} does not bundle the expected MKL DLLs:\n"
            + "\n".join(missing_bundles)
        )

    print(f"{path}: bundles MKL DLLs")


def main():
    parser = argparse.ArgumentParser(
        description="Fail if a Windows wheel does not bundle the MKL DLLs it imports."
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
