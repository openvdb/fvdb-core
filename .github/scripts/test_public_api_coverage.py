# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the public API documentation coverage gate."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXTENSIONS = Path(__file__).resolve().parents[2] / "docs" / "_ext"
SOURCE = '''
def function(value):
    """Return a value."""
    return value

class Widget:
    """A public class."""

    def method(self):
        """A public method."""

    @property
    def value(self):
        """A public property."""
        return 1

    def _internal(self):
        pass
'''
ENTRIES = [
    ".. py:function:: sample_api.function(value)",
    ".. py:class:: sample_api.Widget",
    ".. py:method:: sample_api.Widget.method()",
    ".. py:attribute:: sample_api.Widget.value",
]
INHERITED_SOURCE = '''
import functools

class _Base:
    def inherited(self):
        """Declared on a private base class."""

    def overridden(self):
        """Replaced by the subclass."""

class Derived(_Base):
    """Inherits from a private in-package base."""

    class Nested:
        """A nested public class."""

        def nested_method(self):
            """A method on the nested class."""

    @functools.cached_property
    def cached(self):
        """A cached property."""
        return 1

    def overridden(self):
        """Subclass version."""
'''
INHERITED_ENTRIES = [
    ".. py:class:: sample_api.Derived",
    ".. py:method:: sample_api.Derived.inherited()",
    ".. py:method:: sample_api.Derived.overridden()",
    ".. py:attribute:: sample_api.Derived.cached",
    ".. py:class:: sample_api.Derived.Nested",
    ".. py:method:: sample_api.Derived.Nested.nested_method()",
]
MOCKED_SOURCE = "from ._native import Native, native_function\n"
MOCKED_ENTRIES = [
    ".. py:class:: sample_api.Native",
    ".. py:attribute:: sample_api.Native.field",
    ".. py:function:: sample_api.native_function(value)",
]


def build(
    tmp_path,
    *,
    entries=ENTRIES,
    threshold=100,
    exports=None,
    extra_source="",
    ignore=(),
    cli_threshold=None,
    mock_imports=(),
    runs=1,
):
    """Build a minimal package's reference using the actual Sphinx coverage builder."""
    package = tmp_path / "sample_api"
    package.mkdir()
    (package / "_impl.py").write_text(SOURCE + extra_source, encoding="utf-8")
    if exports is None:
        exports = ["function", "Widget"]
    (package / "__init__.py").write_text(f"from ._impl import *\n__all__ = {exports!r}\n", encoding="utf-8")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "conf.py").write_text(
        f"import sys\nsys.path.insert(0, {str(EXTENSIONS)!r})\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'public_api_coverage']\n"
        f"autodoc_mock_imports = {list(mock_imports)!r}\n"
        "coverage_public_modules = ['sample_api']\n"
        f"coverage_min_percentage = {threshold!r}\n"
        f"coverage_ignore_pyobjects = {list(ignore)!r}\n",
        encoding="utf-8",
    )
    (docs / "index.rst").write_text("API\n===\n\n" + "\n\n".join(entries) + "\n", encoding="utf-8")
    output = tmp_path / "output"
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    overrides = [] if cli_threshold is None else ["-D", f"coverage_min_percentage={cli_threshold}"]
    for _ in range(runs):
        result = subprocess.run(
            [sys.executable, "-m", "sphinx", "-W", "-b", "coverage", *overrides, str(docs), str(output)],
            env=env,
            capture_output=True,
            text=True,
        )
    report = output / "coverage.json"
    return result, json.loads(report.read_text()) if report.exists() else None


def test_reexports_and_properties_are_counted(tmp_path):
    result, report = build(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["total"] == report["documented"] == 4
    assert report["percentage"] == 100


@pytest.mark.parametrize("removed", ENTRIES)
def test_removing_any_public_entry_fails(tmp_path, removed):
    result, report = build(tmp_path, entries=[entry for entry in ENTRIES if entry != removed])
    assert result.returncode != 0
    assert report["total"] == 4
    assert report["percentage"] == 75
    assert len(report["missing"]) == 1


def test_inherited_nested_and_cached_members_are_counted(tmp_path):
    result, report = build(tmp_path, exports=["Derived"], extra_source=INHERITED_SOURCE, entries=INHERITED_ENTRIES)
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["total"] == report["documented"] == len(INHERITED_ENTRIES)


@pytest.mark.parametrize("removed", INHERITED_ENTRIES[1:])
def test_removing_inherited_or_nested_entry_fails(tmp_path, removed):
    result, report = build(
        tmp_path,
        exports=["Derived"],
        extra_source=INHERITED_SOURCE,
        entries=[entry for entry in INHERITED_ENTRIES if entry != removed],
    )
    assert result.returncode != 0
    assert len(report["missing"]) == 1


def test_mocked_imports_are_measured_without_autodoc_directives(tmp_path):
    """The builder must import under autodoc's mock even when no autodoc directive ran first."""
    result, report = build(
        tmp_path,
        exports=["Native", "native_function"],
        extra_source=MOCKED_SOURCE,
        entries=MOCKED_ENTRIES,
        mock_imports=["sample_api._native"],
        runs=2,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["total"] == report["documented"] == 2


def test_mocked_class_without_documented_members_fails(tmp_path):
    result, report = build(
        tmp_path,
        exports=["Native", "native_function"],
        extra_source=MOCKED_SOURCE,
        entries=[entry for entry in MOCKED_ENTRIES if "Native.field" not in entry],
        mock_imports=["sample_api._native"],
    )
    assert result.returncode != 0
    assert "members cannot be enumerated" in result.stderr
    assert report["missing"] == ["sample_api.Native"]


@pytest.mark.parametrize("threshold,passes", [(75, True), (75.01, False), (0, True)])
def test_threshold_boundary(tmp_path, threshold, passes):
    result, report = build(tmp_path, entries=ENTRIES[:-1], threshold=threshold)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    assert report["percentage"] == 75


@pytest.mark.parametrize("threshold,passes", [(75, True), (75.01, False), (0, True)])
def test_cli_threshold_overrides_config(tmp_path, threshold, passes):
    result, report = build(tmp_path, entries=ENTRIES[:-1], threshold=100, cli_threshold=threshold)
    assert (result.returncode == 0) == passes, result.stdout + result.stderr
    assert report["threshold"] == threshold
    assert report["percentage"] == 75


def test_new_export_without_docstring_fails(tmp_path):
    result, report = build(
        tmp_path,
        exports=["function", "Widget", "new_function"],
        extra_source="\ndef new_function():\n    pass\n",
    )
    assert result.returncode != 0
    assert report["missing"] == ["sample_api.new_function"]


def test_explicit_exclusion(tmp_path):
    result, report = build(tmp_path, entries=ENTRIES[:-1], ignore=[r"^sample_api\.Widget\.value$"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["total"] == report["documented"] == 3


@pytest.mark.parametrize("exports", [[], ["does_not_exist"]])
def test_empty_or_invalid_exports_fail(tmp_path, exports):
    result, _ = build(tmp_path, exports=exports)
    assert result.returncode != 0


@pytest.mark.parametrize("threshold", [-1, 101, "nan"])
def test_invalid_threshold_fails(tmp_path, threshold):
    result, _ = build(tmp_path, threshold=threshold)
    assert result.returncode != 0


def test_pytest_skips_sphinx_extensions(tmp_path):
    """Keep Python docstring collection out of the Sphinx-only extension directory."""
    (tmp_path / "pyproject.toml").write_text((EXTENSIONS.parents[1] / "pyproject.toml").read_text(), encoding="utf-8")
    extension_dir = tmp_path / "docs" / "_ext"
    extension_dir.mkdir(parents=True)
    (extension_dir / "extension.py").write_text(
        'raise RuntimeError("Sphinx extension must not be imported by pytest")\n', encoding="utf-8"
    )
    (tmp_path / "docs" / "example.py").write_text('"""\n>>> 1 + 1\n2\n"""\n', encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--doctest-modules", "../docs", "-q"],
        cwd=test_dir,
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
