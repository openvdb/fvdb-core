# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Check the hand-written ``_fvdb_cpp.pyi`` stub against the compiled pybind11 bindings.

``inspect.signature`` cannot read pybind11 functions, so ``mypy.stubtest`` skips their parameters.
pybind11 does put a full signature on the first line of each docstring (one per overload), and that
signature is what this test compares the stub with: names, order, defaults, which parameters accept
``None`` or a JaggedTensor, and the arity of fixed-size tuple returns. Types are not compared
literally, because pybind11 spells ``int`` as ``SupportsInt | SupportsIndex``.
"""

import ast
import inspect
import re
import unittest
from dataclasses import dataclass
from pathlib import Path

import fvdb
from fvdb import _fvdb_cpp

STUB_PATH = Path(fvdb.__file__).parent / "_fvdb_cpp.pyi"

_OPEN = "([{<"
_CLOSE = ")]}>"


@dataclass(frozen=True)
class Param:
    name: str
    annotation: str
    has_default: bool


@dataclass(frozen=True)
class Signature:
    params: tuple[Param, ...]
    returns: str


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    """Split ``text`` on ``sep`` where it is not nested inside brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_binding_param(text: str) -> Param:
    name, _, rest = text.partition(":")
    annotation, eq, _ = rest.partition(" = ")
    return Param(name.strip().lstrip("*"), annotation.strip(), bool(eq))


def binding_signatures(name: str, func) -> list[Signature]:
    """The signatures pybind11 reports for ``func``, one per overload."""
    pattern = re.compile(rf"^(?:\d+\. )?{re.escape(name)}\((.*)\) -> (.*)$")
    signatures = []
    for line in (func.__doc__ or "").splitlines():
        match = pattern.match(line)
        if match is None or match.group(1) == "*args, **kwargs":
            continue
        params = tuple(_parse_binding_param(p) for p in _split_top_level(match.group(1)))
        signatures.append(Signature(params, match.group(2).strip()))
    return signatures


def stub_signatures(tree: ast.Module) -> dict[str, list[Signature]]:
    """Module-level function signatures in the stub, one per ``@overload``."""
    result: dict[str, list[Signature]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        first_default = len(positional) - len(args.defaults)
        params = [
            Param(a.arg, ast.unparse(a.annotation) if a.annotation else "", i >= first_default)
            for i, a in enumerate(positional)
        ]
        if args.vararg:
            params.append(Param(args.vararg.arg, "", False))
        params += [
            Param(a.arg, ast.unparse(a.annotation) if a.annotation else "", d is not None)
            for a, d in zip(args.kwonlyargs, args.kw_defaults)
        ]
        if args.kwarg:
            params.append(Param(args.kwarg.arg, "", False))
        returns = ast.unparse(node.returns) if node.returns else ""
        result.setdefault(node.name, []).append(Signature(tuple(params), returns))
    return result


def _tuple_elements(annotation: str) -> list[str] | None:
    """Elements of a fixed-size ``tuple[...]`` annotation, else ``None``."""
    match = re.fullmatch(r"(?:typing\.)?[Tt]uple\[(.*)\]", annotation.strip())
    if match is None:
        return None
    elements = _split_top_level(match.group(1))
    return None if "..." in elements else elements


def _accepts_none(annotation: str) -> bool:
    return re.search(r"\bNone\b|\bOptional\[", annotation) is not None


def _is_jagged(annotation: str) -> bool:
    return "JaggedTensor" in annotation


def signature_mismatches(binding: Signature, stub: Signature) -> list[str]:
    """Human-readable differences between a binding signature and its stub."""
    names = [p.name for p in binding.params]
    stub_names = [p.name for p in stub.params]
    if names != stub_names:
        return [f"parameters {stub_names} in the stub, {names} in the binding"]
    problems = []
    for b, s in zip(binding.params, stub.params):
        if b.has_default != s.has_default:
            problems.append(f"{b.name}: default {'missing from' if b.has_default else 'extra in'} the stub")
        if _accepts_none(b.annotation) != _accepts_none(s.annotation):
            problems.append(f"{b.name}: binding takes {b.annotation!r}, stub says {s.annotation!r}")
        elif _is_jagged(b.annotation) != _is_jagged(s.annotation):
            problems.append(f"{b.name}: binding takes {b.annotation!r}, stub says {s.annotation!r}")
    # Bound tensors come back as None when undefined, so None-ness of returns is not compared.
    elements = _tuple_elements(binding.returns)
    stub_elements = _tuple_elements(stub.returns)
    if elements is not None:
        if stub_elements is None or len(stub_elements) != len(elements):
            problems.append(f"returns a {len(elements)}-tuple, stub says {stub.returns!r}")
        elif [_is_jagged(e) for e in elements] != [_is_jagged(e) for e in stub_elements]:
            problems.append(f"returns {binding.returns!r}, stub says {stub.returns!r}")
    elif _is_jagged(binding.returns) != _is_jagged(stub.returns):
        problems.append(f"returns {binding.returns!r}, stub says {stub.returns!r}")
    return problems


class FvdbCppStubTests(unittest.TestCase):
    """The module-level functions in ``_fvdb_cpp.pyi`` match the compiled bindings."""

    @classmethod
    def setUpClass(cls):
        cls.stubs = stub_signatures(ast.parse(STUB_PATH.read_text()))
        cls.bindings = {
            name: binding_signatures(name, obj)
            for name, obj in vars(_fvdb_cpp).items()
            if callable(obj) and not inspect.isclass(obj) and not name.startswith("__")
        }

    def test_every_binding_has_a_stub(self):
        missing = sorted(set(self.bindings) - set(self.stubs))
        self.assertEqual(missing, [], "bindings with no stub in _fvdb_cpp.pyi")

    def test_every_stub_has_a_binding(self):
        extra = sorted(set(self.stubs) - set(self.bindings))
        self.assertEqual(extra, [], "stubs in _fvdb_cpp.pyi with no binding")

    def test_bindings_report_signatures(self):
        unparsed = sorted(name for name, signatures in self.bindings.items() if not signatures)
        self.assertEqual(unparsed, [], "bindings whose docstring has no parseable signature")

    def test_stub_signatures_match_bindings(self):
        # Overloads pair up by parameter names, in any order. A stub may cover several bound
        # overloads that differ only in parameter types (e.g. a str or torch.device ``device``).
        problems = []
        for name, signatures in sorted(self.bindings.items()):
            stubs = self.stubs.get(name, [])
            if len(signatures) == 1 and len(stubs) == 1:
                problems += [f"{name}: {p}" for p in signature_mismatches(signatures[0], stubs[0])]
                continue
            for binding in signatures:
                names = [p.name for p in binding.params]
                candidates = [s for s in stubs if [p.name for p in s.params] == names]
                if not candidates:
                    problems.append(f"{name}: no stub overload with parameters {names}")
                    continue
                mismatches = [signature_mismatches(binding, s) for s in candidates]
                if all(mismatches):
                    problems += [f"{name}{names}: {p}" for p in mismatches[0]]
            bound_names = [[p.name for p in b.params] for b in signatures]
            for stub in stubs:
                names = [p.name for p in stub.params]
                if signatures and names not in bound_names:
                    problems.append(f"{name}: stub overload with parameters {names} matches no binding")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
