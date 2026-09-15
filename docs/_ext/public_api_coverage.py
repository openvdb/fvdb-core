# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Adapt Sphinx coverage to re-exported Python APIs and enforce a percentage floor."""

import inspect
import json
import math
from importlib import import_module

from sphinx.errors import SphinxError
from sphinx.ext.coverage import CoverageBuilder
from sphinx.util import logging

logger = logging.getLogger(__name__)


def public_objects(module):
    """Yield exported objects and methods/properties declared on exported classes."""
    exports = getattr(module, "__all__", None)
    if not exports:
        raise SphinxError(f"Coverage module {module.__name__} must declare a nonempty __all__")
    for name in exports:
        obj = getattr(module, name)  # An invalid export must fail the build.
        if inspect.ismodule(obj):
            continue
        full_name = f"{module.__name__}.{name}"
        yield full_name
        if not inspect.isclass(obj):
            continue
        for attr_name, attr in vars(obj).items():
            if attr_name.startswith("_"):
                continue
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            if inspect.isfunction(attr) or isinstance(attr, property):
                yield f"{full_name}.{attr_name}"


class PublicAPICoverageBuilder(CoverageBuilder):
    """Use Sphinx's coverage reports with an export-aware Python object inventory.

    The stock builder skips objects whose defining module differs from their
    public import path, and does not include properties in its denominator.
    Replacing only Python inventory collection retains its report generation.
    """

    def build_py_coverage(self):
        """Compare package exports with objects registered in Sphinx's Python domain."""
        seen = self.env.domaindata["py"]["objects"]
        for module_name in self.config.coverage_public_modules:
            module = import_module(module_name)
            expected = {name for name in public_objects(module) if not self.ignore_pyobj(name)}
            documented = expected.intersection(seen)
            missing = expected - documented
            self.py_documented[module_name] = documented
            self.py_undocumented[module_name] = missing
            # Sphinx's text report groups functions separately from class methods.
            funcs = []
            classes = {}
            for name in sorted(missing):
                relative = name[len(module_name) + 1 :]
                if "." in relative:
                    cls_name, method = relative.split(".", 1)
                    classes.setdefault(cls_name, []).append(method)
                elif inspect.isclass(getattr(module, relative)):
                    classes.setdefault(relative, [])
                else:
                    funcs.append(relative)
            self.py_undoc[module_name] = {"funcs": funcs, "classes": classes}

    def finish(self):
        """Write the standard reports and fail below the configured coverage threshold."""
        super().finish()
        threshold = float(self.config.coverage_min_percentage)
        if not math.isfinite(threshold) or not 0 <= threshold <= 100:
            raise SphinxError("coverage_min_percentage must be between 0 and 100")
        documented = set().union(*self.py_documented.values())
        missing = set().union(*self.py_undocumented.values())
        total = len(documented | missing)
        if not total:
            raise SphinxError("Python API coverage collected no objects")
        percentage = 100 * len(documented) / total
        report = {
            "documented": len(documented),
            "total": total,
            "percentage": percentage,
            "threshold": threshold,
            "missing": sorted(missing),
        }
        (self.outdir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logger.info("Public Python API coverage: %.2f%% (%d/%d)", percentage, len(documented), total)
        if percentage < threshold:
            raise SphinxError(
                f"Public Python API coverage {percentage:.2f}% is below {threshold:g}%; "
                f"see {self.outdir / 'python.txt'} for missing objects"
            )


def setup(app):
    """Register the export-aware coverage builder and configurable coverage floor."""
    app.setup_extension("sphinx.ext.coverage")
    app.add_config_value("coverage_public_modules", [], "env", types=[list])
    app.add_config_value("coverage_min_percentage", 100.0, "env", types=[float, int])
    app.add_builder(PublicAPICoverageBuilder, override=True)
    return {"version": "1", "parallel_read_safe": True}
