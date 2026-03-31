#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from pathlib import Path


def main() -> None:
    tomllib = importlib.import_module("tomllib")
    with Path("pyproject.toml").open("rb") as f:
        viewer_deps = tomllib.load(f)["project"]["optional-dependencies"]["viewer"]

    print(next(dep for dep in viewer_deps if dep.startswith("nanovdb-editor")))


if __name__ == "__main__":
    main()
