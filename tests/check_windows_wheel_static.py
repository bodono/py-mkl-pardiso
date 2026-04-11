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


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
        extension = find_extension(Path(tmp))
        imports = imported_dlls(extension)

    forbidden = [name for name in imports if name.startswith("mkl") and name.endswith(".dll")]
    if forbidden:
        raise RuntimeError(
            f"{path} imports MKL DLLs at runtime:\n" + "\n".join(forbidden)
        )
    print(f"{path}: no MKL DLL imports")


def main():
    parser = argparse.ArgumentParser(
        description="Fail if a Windows wheel depends on MKL DLLs at runtime."
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
