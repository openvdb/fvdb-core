#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Builds a production fvdb-core wheel inside a Docker container from the local
# checkout (uncommitted changes included) and copies it out to --output-dir.
# This is the same recipe the publish workflows use; version defaults come
# from .github/versions.json. Requires only Docker (with BuildKit) and the
# Python standard library.

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VERSIONS_JSON = REPO_ROOT / ".github" / "versions.json"


def die(message):
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def load_versions():
    if not VERSIONS_JSON.is_file():
        die(f"cannot find {VERSIONS_JSON}")
    with open(VERSIONS_JSON) as f:
        return json.load(f)


def parse_args(versions):
    parser = argparse.ArgumentParser(
        description="Builds a production fvdb-core wheel in Docker from this checkout "
        "and copies it to the output directory. Defaults come from .github/versions.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        default=versions["python"]["default"],
        metavar="VER",
        help="Python version (installed via uv, so any version uv can provide)",
    )
    parser.add_argument(
        "--torch",
        dest="torch_version",
        default=versions["torch"]["full_version"],
        metavar="VER",
        help="Full PyTorch version",
    )
    parser.add_argument(
        "--cuda",
        dest="cuda_version",
        default=versions["cuda"]["default"],
        metavar="VER",
        help="CUDA version. Versions listed in .github/versions.json map to their "
        "tested nvidia/cuda image tag; for anything else pass a full "
        "major.minor.patch version matching an nvidia/cuda image tag (e.g. 12.6.3)",
    )
    parser.add_argument(
        "--cuda-arch-list",
        default=versions["cuda"]["arch_list_publish"],
        metavar="LIST",
        help='TORCH_CUDA_ARCH_LIST, or "native" to detect the host GPU via nvidia-smi',
    )
    parser.add_argument(
        "--version-mode",
        choices=["suffix", "nightly", "none"],
        default=None,
        help="How to stamp the wheel version: "
        "suffix = append +pt<torch>.cu<cuda> to the pyproject.toml version (default, publish mode); "
        "nightly = <base>.dev<YYYYMMDD>+pt<torch>.cu<cuda>; "
        "none = leave pyproject.toml version unchanged",
    )
    parser.add_argument(
        "--version",
        dest="version_override",
        default=None,
        metavar="STRING",
        help="Exact version override (mutually exclusive with --version-mode)",
    )
    parser.add_argument(
        "--skip-auditwheel",
        action="store_true",
        help="Skip the auditwheel manylinux repair step",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel build jobs (CMAKE_BUILD_PARALLEL_LEVEL); recommended when "
        "limiting container memory, since auto-detection sees total host RAM",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dist",
        metavar="DIR",
        help="Where the wheel is written",
    )
    args = parser.parse_args()

    if args.version_override is not None and args.version_mode is not None:
        die("--version and --version-mode are mutually exclusive")
    if args.version_mode is None:
        args.version_mode = "suffix"
    if args.jobs is not None and args.jobs < 1:
        die("--jobs must be a positive integer")
    return args


def detect_native_arch_list():
    if shutil.which("nvidia-smi") is None:
        die("--cuda-arch-list native requires nvidia-smi on the host")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    caps = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not caps:
        die("could not detect a GPU compute capability via nvidia-smi")
    unique_caps = list(dict.fromkeys(caps))
    arch_list = ";".join(f"{cap}+PTX" for cap in unique_caps)
    print(f"Detected native CUDA architectures: {arch_list}")
    return arch_list


# PEP 440 version in normalized form (see the "Appendix: Parsing version
# strings with regular expressions" section of the spec). Segment order
# matters: .post must precede .dev.
PEP440_RE = re.compile(
    r"^([1-9][0-9]*!)?"
    r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?"
    r"(\.post(0|[1-9][0-9]*))?"
    r"(\.dev(0|[1-9][0-9]*))?"
    r"(\+[a-z0-9]+(\.[a-z0-9]+)*)?$"
)


def read_pyproject_version():
    for line in (REPO_ROOT / "pyproject.toml").read_text().splitlines():
        match = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if match:
            return match.group(1)
    die("failed to read version from pyproject.toml")


def compute_wheel_version(args, local_suffix):
    """Return the version to stamp into pyproject.toml, or "" to leave it unchanged."""
    current = read_pyproject_version()
    if args.version_override is not None:
        return current, args.version_override
    if args.version_mode == "suffix":
        return current, f"{current}{local_suffix}"
    if args.version_mode == "nightly":
        # Anchor the nightly at the upcoming release recorded in pyproject.toml
        # (e.g. 0.6.0.dev0 -> 0.6.0) so PEP 440 ordering puts nightlies between
        # the previous final release and the next one.
        base = re.sub(r"(\.dev[0-9]+|\.post[0-9]+|(a|b|c|rc)[0-9]+)+$", "", current.split("+")[0])
        if not base:
            die(f"failed to parse base version from pyproject.toml (got '{current}')")
        date_stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
        return current, f"{base}.dev{date_stamp}{local_suffix}"
    return current, ""


def main():
    versions = load_versions()
    args = parse_args(versions)

    if shutil.which("docker") is None:
        die("docker is required but not found")
    buildx = subprocess.run(["docker", "buildx", "version"], capture_output=True)
    if buildx.returncode != 0:
        die("docker with BuildKit (buildx) is required; " "upgrade Docker or install the buildx plugin")

    known_cuda = versions["cuda"]["versions"].get(args.cuda_version)
    if known_cuda is not None:
        cuda_image_tag = known_cuda["patch"]
    else:
        # Untested CUDA version: use it verbatim as the nvidia/cuda image tag,
        # which requires the full major.minor.patch form.
        cuda_image_tag = args.cuda_version
        print(
            f"Note: CUDA {args.cuda_version} is not in .github/versions.json; "
            f"using nvidia/cuda:{cuda_image_tag}-cudnn-devel-rockylinux8 directly"
        )
    cuda_components = args.cuda_version.split(".")
    cuda_tag = "cu" + "".join(cuda_components[:2])
    cuda_major = cuda_components[0]
    torch_tag = "".join(args.torch_version.split(".")[:2])
    local_suffix = f"+pt{torch_tag}.{cuda_tag}"

    cuda_arch_list = args.cuda_arch_list
    if cuda_arch_list == "native":
        cuda_arch_list = detect_native_arch_list()

    auditwheel_excludes = [f"--exclude {lib}" for lib in versions["auditwheel_excludes"]]
    auditwheel_excludes += [f"--exclude {lib}.{cuda_major}" for lib in versions["auditwheel_excludes_cuda_major"]]

    current_version, wheel_version = compute_wheel_version(args, local_suffix)
    if wheel_version and not PEP440_RE.match(wheel_version):
        die(
            f"'{wheel_version}' is not a valid normalized PEP 440 version; "
            "the build would fail when scikit-build-core parses pyproject.toml "
            "(note PEP 440 puts .post before .dev, e.g. 1.2.3.post1.dev0)"
        )

    print("Building fvdb-core wheel with:")
    print(f"  Python:          {args.python_version}")
    print(f"  PyTorch:         {args.torch_version} " f"(index: https://download.pytorch.org/whl/{cuda_tag})")
    print(f"  CUDA:            {args.cuda_version} " f"(image: nvidia/cuda:{cuda_image_tag}-cudnn-devel-rockylinux8)")
    print(f"  CUDA archs:      {cuda_arch_list}")
    print(f"  Wheel version:   {wheel_version or f'{current_version} (unchanged)'}")
    print(f"  auditwheel:      {'skipped' if args.skip_auditwheel else 'enabled'}")
    print(f"  Build jobs:      {args.jobs if args.jobs is not None else 'auto'}")
    print(f"  Output dir:      {args.output_dir}")
    print()

    build_args = {
        "CUDA_IMAGE_TAG": cuda_image_tag,
        "UV_VERSION": versions["uv"]["version"],
        "PYTHON_VERSION": args.python_version,
        "TORCH_VERSION": args.torch_version,
        "CUDA_TAG": cuda_tag,
        "CUDA_ARCH_LIST": cuda_arch_list,
        "GCC_TOOLSET": versions["gcc"]["toolset"],
        "CMAKE_VERSION": versions["cmake_version"],
        "WHEEL_VERSION": wheel_version,
        "RUN_AUDITWHEEL": "0" if args.skip_auditwheel else "1",
        "AUDITWHEEL_EXCLUDES": " ".join(auditwheel_excludes),
        "BUILD_JOBS": str(args.jobs) if args.jobs is not None else "",
    }
    command = [
        "docker",
        "build",
        "--file",
        str(SCRIPT_DIR / "Dockerfile.wheel"),
        "--target",
        "export",
        "--output",
        f"type=local,dest={args.output_dir}",
    ]
    for name, value in build_args.items():
        command += ["--build-arg", f"{name}={value}"]
    command.append(str(REPO_ROOT))

    result = subprocess.run(command, env={**os.environ, "DOCKER_BUILDKIT": "1"})
    if result.returncode != 0:
        sys.exit(result.returncode)

    wheels = sorted(args.output_dir.glob("fvdb_core-*.whl"))
    print()
    print(f"Wheel(s) written to {args.output_dir}:")
    for wheel in wheels:
        print(f"  {wheel}")
    print()
    print("Install with the matching PyTorch build, e.g.:")
    print(
        f"  pip install torch=={args.torch_version} " f"--extra-index-url https://download.pytorch.org/whl/{cuda_tag}"
    )
    print(f"  pip install {args.output_dir}/fvdb_core-*.whl")


if __name__ == "__main__":
    main()
