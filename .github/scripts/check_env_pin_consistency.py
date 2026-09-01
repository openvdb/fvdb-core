#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Repo-specific CI policy for toolchain pins shared across the conda envs.

``env/`` holds several conda environments (build, dev, test, learn, ...) that
each pin the toolchain independently. Nothing has kept them in agreement, yet a
disagreement is not cosmetic:

  * ``build_environment.yml`` is the environment the wheel is compiled in, so it
    determines the resulting ABI.
  * A developer working from ``dev_environment.yml`` with a different PyTorch
    would build or load a wheel against a mismatched libtorch and hit an
    undefined-symbol ImportError.
  * fvdb-reality-capture derives its benchmark environment from
    ``build_environment.yml``; if the env files here disagree, "which PyTorch
    does fvdb-core use?" has no single answer.

The rule enforced here is deliberately permissive about *presence* and strict
about *agreement*:

    If two environment files pin the same key, they must pin the same value.

A file that omits a key is fine (``release_base_environment.yml`` pins none of
them), so adding a slimmer environment does not require touching this script.

Run ``--fix`` to propagate the canonical file's values to the others.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Keys that affect the compiled ABI or the interpreter the wheel targets.
PINNED_KEYS = ("python", "cuda-version", "pytorch-gpu")

# The environment the wheel is actually built in, hence the source of truth for
# --fix and the file downstream repositories derive from.
CANONICAL_ENV = "build_environment.yml"


def pin_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"^(?P<prefix>\s*-\s*{re.escape(key)}=)(?P<value>\S+)\s*$", re.MULTILINE)


def read_pins(text: str) -> dict[str, str]:
    """Return only the keys this file actually pins."""
    pins = {}
    for key in PINNED_KEYS:
        match = pin_pattern(key).search(text)
        if match is not None:
            pins[key] = match.group("value")
    return pins


def collect(env_dir: Path) -> dict[Path, dict[str, str]]:
    paths = sorted(p for p in env_dir.glob("*.yml"))
    if not paths:
        raise SystemExit(f"error: no environment files found in {env_dir}")
    return {path: read_pins(path.read_text(encoding="utf-8")) for path in paths}


def find_disagreements(pins_by_file: dict[Path, dict[str, str]]) -> dict[str, dict[str, list[Path]]]:
    """Map key -> {value -> files pinning it}, for keys pinned inconsistently."""
    disagreements: dict[str, dict[str, list[Path]]] = {}
    for key in PINNED_KEYS:
        by_value: dict[str, list[Path]] = defaultdict(list)
        for path, pins in pins_by_file.items():
            if key in pins:
                by_value[pins[key]].append(path)
        if len(by_value) > 1:
            disagreements[key] = dict(by_value)
    return disagreements


def apply_fix(pins_by_file: dict[Path, dict[str, str]], canonical: Path) -> list[Path]:
    canonical_pins = pins_by_file[canonical]
    changed = []
    for path, pins in pins_by_file.items():
        if path == canonical:
            continue
        text = original = path.read_text(encoding="utf-8")
        for key, value in canonical_pins.items():
            if key in pins and pins[key] != value:
                text = pin_pattern(key).sub(lambda m: f"{m.group('prefix')}{value}", text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("env_dir", nargs="?", default="env", type=Path, help="directory of conda env files")
    parser.add_argument("--fix", action="store_true", help=f"propagate {CANONICAL_ENV}'s values to the other files")
    args = parser.parse_args(argv)

    env_dir = args.env_dir
    if not env_dir.is_dir():
        raise SystemExit(f"error: {env_dir} is not a directory")

    pins_by_file = collect(env_dir)
    canonical = env_dir / CANONICAL_ENV
    if canonical not in pins_by_file:
        raise SystemExit(f"error: canonical environment {canonical} not found")

    for path in sorted(pins_by_file):
        pins = pins_by_file[path]
        rendered = "  ".join(f"{key}={pins[key]}" for key in PINNED_KEYS if key in pins) or "(pins none)"
        print(f"  {path.name:<32} {rendered}")

    if args.fix:
        changed = apply_fix(pins_by_file, canonical)
        if not changed:
            print(f"\nAll environment files already agree with {CANONICAL_ENV}.")
            return 0
        print(f"\nUpdated from {CANONICAL_ENV}: {', '.join(p.name for p in changed)}")
        return 0

    disagreements = find_disagreements(pins_by_file)
    if not disagreements:
        print("\nAll environment files agree on the shared toolchain pins.")
        return 0

    print("", file=sys.stderr)
    for key, by_value in disagreements.items():
        print(f"error: '{key}' is pinned inconsistently across env/:", file=sys.stderr)
        for value, paths in sorted(by_value.items()):
            print(f"    {value:<12} {', '.join(p.name for p in paths)}", file=sys.stderr)
    print(
        "\nThese environments share a toolchain: a mismatch means a wheel built in one\n"
        "cannot be loaded in another (undefined libtorch symbols at `import fvdb`).\n"
        f"Fix by editing {CANONICAL_ENV} and running:\n"
        "    python3 .github/scripts/check_env_pin_consistency.py --fix",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
