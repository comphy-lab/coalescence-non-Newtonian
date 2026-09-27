"""The two solvers stay independent: they share only case files.

Agreement between the boundary-integral and finite-element solvers is evidence about
their numerics only if neither borrows code from the other.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "coalescence"


def imported_modules(package: Path) -> set[str]:
    names = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names.add(node.module)
    return names


class LayoutTests(unittest.TestCase):
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
