# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    parser.addoption(
        "--doc-dirs",
        action="append",
        default=None,
        help="Directories of markdown files to scan for API ref validation "
        "(relative to repo root). Repeatable. Default: docs/TEACHME, docs/tutorials",
    )
