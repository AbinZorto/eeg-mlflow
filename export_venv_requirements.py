#!/usr/bin/env python3
import pathlib
import sys

def collect_requirements(venv_path: pathlib.Path):
    site_dirs = sorted((venv_path / "lib").glob("python*/site-packages"))
    if not site_dirs:
        raise SystemExit(f"No site-packages found under {venv_path}/lib")

    reqs = set()
    site = site_dirs[-1]  # pick highest pythonX.Y
    for dist_info in site.glob("*.dist-info"):
        meta = dist_info / "METADATA"
        name = version = None

        if meta.exists():
            for line in meta.read_text(errors="ignore").splitlines():
                if line.startswith("Name: "):
                    name = line[6:].strip()
                elif line.startswith("Version: "):
                    version = line[9:].strip()
                if name and version:
                    break

        if not (name and version):
            base = dist_info.name[:-10]  # strip .dist-info
            parts = base.split("-")
            if len(parts) >= 2:
                name, version = "-".join(parts[:-1]), parts[-1]

        if name and version:
            reqs.add(f"{name}=={version}")

    return sorted(reqs, key=str.lower)


def main():
    venv = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".venv")
    if not venv.is_dir():
        raise SystemExit(f"Venv not found: {venv}")
    for line in collect_requirements(venv):
        print(line)


if __name__ == "__main__":
    main()
