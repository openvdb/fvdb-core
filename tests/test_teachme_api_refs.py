#!/usr/bin/env python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""
Validate that API references in markdown doc files match the actual fvdb API.

Checks:
1. Inline backtick references in prose (outside fenced code blocks).
2. fvdb API calls inside ``notest`` code blocks (to prevent rot).

Classes and their attributes are auto-discovered by introspecting the fvdb
module — no hardcoded maps to maintain.

Usage::

    # Scan default dirs (docs/TEACHME, docs/tutorials)
    pytest tests/test_teachme_api_refs.py -v

    # Scan specific dirs
    pytest tests/test_teachme_api_refs.py --doc-dirs docs/tutorials -v

    # Scan multiple dirs
    pytest tests/test_teachme_api_refs.py --doc-dirs docs/TEACHME --doc-dirs docs/tutorials -v
"""

import inspect
import re
from pathlib import Path

import pytest

import fvdb
import fvdb.nn

_ROOT = Path(__file__).parent.parent
_DEFAULT_DOC_DIRS = ["docs/TEACHME", "docs/tutorials"]

# ---------------------------------------------------------------------------
# Auto-discover fvdb's API surface
# ---------------------------------------------------------------------------
_MODULES = [fvdb, fvdb.nn]

# class name -> class object (e.g. "GridBatch" -> fvdb.GridBatch)
_CLASSES: dict[str, type] = {}
for _mod in _MODULES:
    for _name, _obj in inspect.getmembers(_mod, inspect.isclass):
        if not _name.startswith("_"):
            _CLASSES[_name] = _obj

# All known attribute names across all fvdb classes (for instance.attr checks)
_ALL_ATTRS: set[str] = set()
for _cls in _CLASSES.values():
    _ALL_ATTRS.update(a for a in dir(_cls) if not a.startswith("_"))

# All known top-level names in fvdb and fvdb.nn
_TOP_LEVEL: set[str] = set()
for _mod in _MODULES:
    _TOP_LEVEL.update(a for a in dir(_mod) if not a.startswith("_"))

# References that should be skipped (wildcards, third-party, etc.)
_SKIP_PREFIXES = ("torch.", "optimizer.", "pcu.", "np.", "loss.")
_SKIP_EXACT = {"fvdb.*", "fvdb.viz.*Splat*", "GridBatch.from_*"}


def _get_scan_dirs(config) -> list[Path]:
    """Resolve --doc-dirs option to absolute Paths.

    Raises pytest.UsageError if user-supplied dirs don't exist or if
    no valid directories remain (prevents silent no-op test runs).
    """
    raw = config.getoption("doc_dirs")
    user_supplied = bool(raw)
    raw = raw or _DEFAULT_DOC_DIRS

    dirs = []
    missing = []
    for d in raw:
        p = Path(d) if Path(d).is_absolute() else _ROOT / d
        if p.is_dir():
            dirs.append(p)
        else:
            missing.append(f"{d} -> {p}")

    if user_supplied and missing:
        raise pytest.UsageError(
            "Invalid --doc-dirs value(s): "
            + ", ".join(missing)
            + ". Each --doc-dirs path must exist and be a directory."
        )

    if not dirs:
        source = "--doc-dirs" if user_supplied else "default doc directories"
        raise pytest.UsageError(
            f"No documentation directories to scan from {source}: " + ", ".join(str(d) for d in raw)
        )

    return dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_call_parens(ref: str) -> str:
    """Remove trailing call syntax: 'foo.bar(x, y)' -> 'foo.bar'."""
    match = re.match(r"^([\w.]+)", ref)
    return match.group(1) if match else ref


def _is_skip(ref: str, clean: str) -> bool:
    """Return True if this reference should not be validated."""
    if ref in _SKIP_EXACT:
        return True
    if any(clean.startswith(p) for p in _SKIP_PREFIXES):
        return True
    # File paths, notebooks
    if "/" in ref or ref.endswith((".ipynb", ".py", ".md")):
        return True
    # Numeric values
    if ref.replace(".", "").replace("-", "").replace(" ", "").isdigit():
        return True
    # Expressions with operators
    if " = " in ref or " > " in ref or " < " in ref:
        return True
    return False


def _looks_like_api_ref(clean: str) -> bool:
    """Return True if the reference looks like an fvdb API reference."""
    parts = clean.split(".")
    if parts[0] in ("fvdb", "fvnn"):
        return True
    if parts[0] in _CLASSES:
        return True
    # instance.attr where attr is a known fvdb class attribute
    if len(parts) >= 2 and parts[1] in _ALL_ATTRS:
        return True
    # Bare class name
    if len(parts) == 1 and parts[0] in _CLASSES:
        return True
    return False


def _resolve(clean: str) -> tuple[bool, str]:
    """Try to resolve a cleaned API reference. Returns (ok, reason)."""
    parts = clean.split(".")

    # fvdb.X.Y.Z — walk the module tree
    if parts[0] == "fvdb":
        obj = fvdb
        for part in parts[1:]:
            if not hasattr(obj, part):
                return False, f"cannot resolve '{clean}' from fvdb module"
            obj = getattr(obj, part)
        return True, "fvdb_path"

    # fvnn.X — check fvdb.nn
    if parts[0] == "fvnn":
        if len(parts) >= 2 and hasattr(fvdb.nn, parts[1]):
            return True, "fvnn_attr"
        return False, f"cannot resolve '{clean}' from fvdb.nn"

    # ClassName or ClassName.method
    if parts[0] in _CLASSES:
        if len(parts) == 1:
            return True, "class_name"
        cls = _CLASSES[parts[0]]
        if hasattr(cls, parts[1]):
            return True, "class_attr"
        return False, f"'{parts[0]}' has no attribute '{parts[1]}'"

    # instance.attr — check if attr exists on any fvdb class
    if len(parts) >= 2 and parts[1] in _ALL_ATTRS:
        return True, "instance_attr"
    if len(parts) >= 2 and parts[1] not in _ALL_ATTRS:
        return False, f"'{parts[1]}' not found on any fvdb class"

    return True, "not_api"


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
def _extract_inline_refs(text: str) -> list[tuple[int, str]]:
    """Extract inline backtick code from markdown, excluding fenced blocks."""
    results = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in re.finditer(r"`([^`]+)`", line):
            results.append((lineno, match.group(1)))
    return results


def _extract_notest_api_refs(text: str) -> list[tuple[int, str]]:
    """Extract fvdb API references from notest code blocks.

    Catches:
    - fvdb.X.Y and fvnn.X.Y module-path references
    - instance.method_name(...) where method_name is a known fvdb class attribute
    """
    results = []
    in_notest = False
    seen = set()  # deduplicate per-file
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```python notest"):
            in_notest = True
            continue
        if stripped.startswith("```") and in_notest:
            in_notest = False
            continue
        if not in_notest:
            continue
        # Match fvdb.X.Y or fvnn.X.Y dotted references in code
        for match in re.finditer(r"\b(fvdb\.[\w.]+|fvnn\.[\w.]+)", line):
            ref = match.group(1).rstrip(".")
            if ref and not _is_skip(ref, ref) and ref not in seen:
                results.append((lineno, ref))
                seen.add(ref)
        # Match instance.method(...) where method is a known fvdb attribute
        for match in re.finditer(r"\b\w+\.([\w]+)\s*[\(=]", line):
            attr = match.group(1)
            if attr in _ALL_ATTRS and attr not in seen:
                full = match.group(0).split(".")[0] + "." + attr
                results.append((lineno, full))
                seen.add(attr)
    return results


# ---------------------------------------------------------------------------
# Gatherers (take scan_dirs as argument)
# ---------------------------------------------------------------------------
def _gather_inline_params(scan_dirs: list[Path]) -> list[tuple[str, int, str]]:
    """Collect all testable inline API references from markdown files."""
    params = []
    for scan_dir in scan_dirs:
        for md_file in sorted(scan_dir.glob("*.md")):
            text = md_file.read_text()
            for lineno, ref in _extract_inline_refs(text):
                clean = _strip_call_parens(ref.strip())
                if _is_skip(ref, clean):
                    continue
                if _looks_like_api_ref(clean):
                    params.append((md_file.name, lineno, ref))
    return params


def _gather_notest_params(scan_dirs: list[Path]) -> list[tuple[str, int, str]]:
    """Collect fvdb API references from notest code blocks."""
    params = []
    for scan_dir in scan_dirs:
        for md_file in sorted(scan_dir.glob("*.md")):
            text = md_file.read_text()
            for lineno, ref in _extract_notest_api_refs(text):
                clean = _strip_call_parens(ref.strip())
                if _looks_like_api_ref(clean):
                    params.append((md_file.name, lineno, ref))
    return params


# ---------------------------------------------------------------------------
# Tests — parametrized at collection time via pytest_generate_tests
# ---------------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    scan_dirs = _get_scan_dirs(metafunc.config)

    if metafunc.function is test_api_ref:
        params = _gather_inline_params(scan_dirs)
        metafunc.parametrize(
            "filename,lineno,ref",
            params,
            ids=[f"{p[0]}:{p[1]}:{p[2][:50]}" for p in params],
        )

    elif metafunc.function is test_notest_api_ref:
        params = _gather_notest_params(scan_dirs)
        metafunc.parametrize(
            "filename,lineno,ref",
            params,
            ids=[f"{p[0]}:{p[1]}:{p[2][:50]}" for p in params],
        )


def test_api_ref(filename: str, lineno: int, ref: str):
    """Each inline API reference in prose should resolve against the fvdb API."""
    clean = _strip_call_parens(ref.strip())
    valid, reason = _resolve(clean)
    assert valid, f"{filename}:{lineno}: {reason}  (ref: `{ref}`)"


def test_notest_api_ref(filename: str, lineno: int, ref: str):
    """API references in notest code blocks should still resolve (prevents rot)."""
    clean = _strip_call_parens(ref.strip())
    valid, reason = _resolve(clean)
    assert valid, f"{filename}:{lineno} (notest block): {reason}  (ref: `{ref}`)"
