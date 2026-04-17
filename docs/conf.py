# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

_versions_path = os.path.join(os.path.dirname(__file__), "..", ".github", "versions.json")
try:
    with open(_versions_path) as _f:
        _versions = json.load(_f)
except FileNotFoundError:
    _versions = {
        "torch": {"full_version": "unknown", "version": "0"},
        "cuda": {"versions": {}},
        "python": {"matrix": ["3.12"]},
    }


# -- Project information -----------------------------------------------------

project = "ƒVDB"
copyright = "Contributors to the OpenVDB Project"
author = "Contributors to the OpenVDB Project"

# Stable fvdb-core version shown in installation examples.
fvdb_core_stable_version = "0.4.2"

version = fvdb_core_stable_version
release = fvdb_core_stable_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3

# Fix return-type in google-style docstrings
napoleon_custom_sections = [("Returns", "params_style")]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = [".rst", ".md"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "wip", "tutorials"]

autodoc_default_options = {"undoc-members": "forward, extra_repr"}

# Mock the compiled C++ extension so Sphinx can introspect the Python API
# on build hosts that lack CUDA (e.g. Read the Docs).
autodoc_mock_imports = ["_fvdb_cpp", "fvdb._fvdb_cpp"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {"analytics_id": "G-60P7VJJ09C"}  # Google Analytics ID

html_context = {
    "display_github": True,
    "github_user": "openvdb",
    "github_repo": "fvdb-core",
    "github_version": "release/v0.4",
    "conf_py_path": "/docs/",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["imgs"]
html_css_files = [
    "css/custom.css",
]


# -- Custom hooks ------------------------------------------------------------


def process_signature(app, what, name, obj, options, signature, return_annotation):
    if signature is not None:
        signature = signature.replace("._fvdb_cpp", "")
        signature = signature.replace("fvdb::", "fvdb.")

    if return_annotation is not None:
        return_annotation = return_annotation.replace("._fvdb_cpp", "")
        return_annotation = return_annotation.replace("fvdb::", "fvdb.")

    return signature, return_annotation


def setup(app):
    pass
    # app.connect("autodoc-process-signature", process_signature)
