# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import glob
import os
import os.path
import shutil
import sys

torch_version = sys.argv[1]
cuda_version = sys.argv[2]
wheel_glob = sys.argv[3] if len(sys.argv) > 3 else "dist/*.whl"

"""
Utility to append a PEP 440 local version to built wheel filenames for offline distribution.

Note: This only renames files on disk and does not modify the wheel's embedded METADATA
version. Do not use this for uploads to PyPI/TestPyPI, as the filename and METADATA
versions must match for a valid upload.
"""

wheels = glob.glob(wheel_glob)
for wheel in wheels:
    wheel = os.path.basename(wheel)
    filename, ext = os.path.splitext(wheel)
    tags = filename.split("-")
    # tags layout: [name, version, build? (optional), pyTag, abiTag, platTag]
    # We expect version at index -4 for standard wheels without build tag
    local_tag = "torch" + torch_version + "." + cuda_version
    new_version = tags[-4] + "+" + local_tag
    new_filename = "-".join(tags[:-4] + [new_version] + tags[-3:]) + ext
    print(f"Renaming {wheel} -> {new_filename}")
    os.rename(os.path.join("dist", wheel), os.path.join("dist", new_filename))
