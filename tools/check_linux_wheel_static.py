import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[(.+)\]")


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


def imported_sos(path):
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


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
        extension = find_extension(Path(tmp))
        imports = imported_sos(extension)

    forbidden = [name for name in imports if name.startswith("libmkl") and ".so" in name]
    if forbidden:
        raise RuntimeError(
            f"{path} imports MKL shared libraries at runtime:\n" + "\n".join(forbidden)
        )
    print(f"{path}: no MKL shared library imports")


def main():
    parser = argparse.ArgumentParser(
        description="Fail if a Linux wheel depends on MKL shared libraries at runtime."
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
