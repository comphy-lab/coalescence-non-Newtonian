"""The two solvers stay independent: they share only case files.

Agreement between the boundary-integral and finite-element solvers is evidence about
their numerics only if neither borrows code from the other.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "coalescence"


def module_imports(source: str, module: str, is_package: bool = False) -> set[str]:
    """Absolute names imported by ``module``, with relative imports resolved against it."""
    package = module.split(".") if is_package else module.split(".")[:-1]
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            prefix = ".".join(base + ([node.module] if node.module else []))
            names.add(prefix)
            names.update(f"{prefix}.{alias.name}" for alias in node.names)
    return names


def imported_modules(package: Path) -> set[str]:
    names = set()
    for path in package.rglob("*.py"):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        names |= module_imports(path.read_text(), module, is_package)
    return names


class LayoutTests(unittest.TestCase):
    def test_relative_imports_are_resolved(self) -> None:
        names = module_imports("from ..fem import q2_geometry\nfrom . import kernels\n", "coalescence.bim.solver")
        self.assertIn("coalescence.fem.q2_geometry", names)
        self.assertIn("coalescence.bim.kernels", names)
        self.assertIn("coalescence.fem", module_imports("from .. import fem\n", "coalescence.bim", is_package=True))

    def test_bim_imports_neither_fem_nor_pyoomph(self) -> None:
        bad = {m for m in imported_modules(SRC / "bim") if m.startswith(("coalescence.fem", "pyoomph"))}
        self.assertEqual(bad, set())

    def test_fem_does_not_import_bim(self) -> None:
        bad = {m for m in imported_modules(SRC / "fem") if m.startswith("coalescence.bim")}
        self.assertEqual(bad, set())

    def test_analysis_imports_no_solver(self) -> None:
        bad = {m for m in imported_modules(SRC / "analysis")
               if m.startswith(("coalescence.fem", "coalescence.bim", "pyoomph"))}
        self.assertEqual(bad, set())


if __name__ == "__main__":
    unittest.main()
