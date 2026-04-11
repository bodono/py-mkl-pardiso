import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


NEEDED_RE = re.compile(r"\(NEEDED\)\s+Shared library: \[(.+)\]")
EXPECTED_MKL_LIBS = (
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


def check_wheel(path):
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(path) as wheel:
            wheel.extractall(tmp)
            wheel_files = [Path(name).name.lower() for name in wheel.namelist()]
        extension = find_extension(Path(tmp))
        imports = imported_shared_libs(extension)

    missing_imports = [
        lib for lib in EXPECTED_MKL_LIBS
        if not any(name.startswith(lib) and ".so" in name for name in imports)
    ]
    if missing_imports:
        raise RuntimeError(
            f"{path} is missing dynamic MKL dependencies:\n" + "\n".join(missing_imports)
        )

    missing_bundles = [
        lib for lib in EXPECTED_MKL_LIBS
        if not any(name.startswith(lib) and ".so" in name for name in wheel_files)
    ]
    if missing_bundles:
        raise RuntimeError(
            f"{path} does not bundle the expected MKL shared libraries:\n"
            + "\n".join(missing_bundles)
        )

    print(f"{path}: bundles MKL shared libraries")


def main():
    parser = argparse.ArgumentParser(
        description="Fail if a Linux wheel does not bundle the MKL shared libraries it imports."
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
