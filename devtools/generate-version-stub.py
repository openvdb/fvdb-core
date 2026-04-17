# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Generate fvdb/version.py from the .in template when CMake has not run.

Used by Read the Docs (see .readthedocs.yaml) where the C++ build step is
skipped.  The script is a no-op if fvdb/version.py already exists.
"""

import pathlib
import sys

version_py = pathlib.Path("fvdb/version.py")
if version_py.exists():
    sys.exit(0)

import tomllib

v = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]["version"]
template = pathlib.Path("fvdb/version.py.in").read_text()
version_py.write_text(template.replace("@SKBUILD_PROJECT_VERSION@", v).replace("@FVDB_GIT_SHA@", "unknown"))
