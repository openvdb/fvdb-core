# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Adapt Sphinx coverage to re-exported Python APIs and enforce a percentage floor."""

import functools
import inspect
import json
import math
from importlib import import_module

from sphinx.errors import SphinxError
from sphinx.ext.autodoc.mock import mock
from sphinx.ext.coverage import CoverageBuilder
from sphinx.util import logging

logger = logging.getLogger(__name__)


def _in_package(obj, package):
    return getattr(obj, "__module__", "").split(".")[0] == package


def _is_mocked(obj):
    return getattr(obj, "__sphinx_mock__", False)


def _class_members(full_name, cls, package):
    """Yield public members declared on ``cls`` or inherited from bases in the same package."""
    shadowed = set()
    for klass in inspect.getmro(cls):
        if not _in_package(klass, package):
            continue
        for attr_name, attr in vars(klass).items():
            if attr_name in shadowed:
                continue
            shadowed.add(attr_name)
            if attr_name.startswith("_"):
                continue
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            member = f"{full_name}.{attr_name}"
            if inspect.isfunction(attr) or isinstance(attr, (property, functools.cached_property)):
                yield member
            elif inspect.isclass(attr) and attr.__qualname__ == f"{klass.__qualname__}.{attr_name}":
                yield member
                yield from _class_members(member, attr, package)


def public_objects(module):
    """Yield exported objects and methods/properties declared on exported classes."""
    exports = getattr(module, "__all__", None)
    if not exports:
        raise SphinxError(f"Coverage module {module.__name__} must declare a nonempty __all__")
    package = module.__name__.split(".")[0]
    for name in exports:
        obj = getattr(module, name)  # An invalid export must fail the build.
        if inspect.ismodule(obj):
            continue
        full_name = f"{module.__name__}.{name}"
        yield full_name
        if inspect.isclass(obj):
            yield from _class_members(full_name, obj, package)


class PublicAPICoverageBuilder(CoverageBuilder):
    """Use Sphinx's coverage reports with an export-aware Python object inventory.

    The stock builder skips objects whose defining module differs from their
    public import path, and does not include properties in its denominator.
    Replacing only Python inventory collection retains its report generation.
    """

    def build_py_coverage(self):
        """Compare package exports with objects registered in Sphinx's Python domain."""
        seen = self.env.domaindata["py"]["objects"]
        mock_imports = getattr(self.config, "autodoc_mock_imports", [])
        for module_name in self.config.coverage_public_modules:
            with mock(mock_imports):
                module = import_module(module_name)
                expected = {name for name in public_objects(module) if not self.ignore_pyobj(name)}
                documented = expected.intersection(seen)
                opaque = self._opaque_classes(module, documented, seen)
                documented -= opaque
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
                elif name in opaque or inspect.isclass(getattr(module, relative)):
                    classes.setdefault(relative, [])
                else:
                    funcs.append(relative)
            self.py_undoc[module_name] = {"funcs": funcs, "classes": classes}

    @staticmethod
    def _opaque_classes(module, documented, seen):
        """Return mocked exports documented as classes without any documented member.

        Members of mocked (compiled) exports cannot be enumerated, so their
        manual reference entries are the only record of the class contents.
        """
        opaque = set()
        for export in module.__all__:
            name = f"{module.__name__}.{export}"
            if name not in documented or seen[name].objtype != "class" or not _is_mocked(getattr(module, export)):
                continue
            if not any(other.startswith(name + ".") for other in seen):
                logger.warning(
                    "%s is a compiled export whose members cannot be enumerated; "
                    "document them manually below its py:class entry",
                    name,
                )
                opaque.add(name)
        return opaque

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
    # Sphinx cannot parse -D overrides for a float default. Accept strings here
    # and convert/validate in finish(), preserving fractional thresholds.
    app.add_config_value("coverage_min_percentage", "100", "env", types=[str, float, int])
    app.add_builder(PublicAPICoverageBuilder, override=True)
    return {"version": "1", "parallel_read_safe": True}
