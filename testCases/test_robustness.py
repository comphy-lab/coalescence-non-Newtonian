"""Case validation, run provenance and refusal paths of the runners, which need no solver run."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STOKES_CASE = ROOT / "validationCases" / "anthony2020" / "T5-stokes-R0-1e-06.json"

from coalescence.cases import case_problems, validate_case  # noqa: E402
from coalescence.provenance import component_commit, pyoomph_revision  # noqa: E402


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaseValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [json.loads(p.read_text()) for p in
                      (ROOT / "simulationCases" / "coalescence.json", *sorted((ROOT / "validationCases").rglob("*.json")))
                      if json.loads(p.read_text()).get("schema") == "pyoomph-case-v1"]

    def test_repository_cases_are_valid(self) -> None:
        self.assertGreaterEqual(len(self.cases), 3)
        for case in self.cases:
            self.assertEqual(case_problems(case), [], case["case_id"])

    def test_each_defect_is_named(self) -> None:
        inertial = next(c for c in self.cases if c["physics"]["inertia"])
        defects = {
            "schema": lambda c: c.update(schema="other"),
            "Oh": lambda c: c["physics"].update(Oh=-1.0),
            "R0": lambda c: c["physics"]["initial_bridge"].update(R0=2.0),
            "Z0": lambda c: c["physics"]["initial_bridge"].pop("Z0"),
            "exterior": lambda c: c["physics"]["exterior"].update(viscosity=0.1),
            "boolean exterior": lambda c: c["physics"]["exterior"].update(density=False),
            "rest": lambda c: c["physics"].update(initial_velocity="uniform"),
        }
        for name, spoil in defects.items():
            case = copy.deepcopy(inertial)
            spoil(case)
            with self.subTest(defect=name), self.assertRaises(SystemExit):
                validate_case(case)

    def test_stokes_only_solver_refuses_finite_oh(self) -> None:
        inertial = next(c for c in self.cases if c["physics"]["inertia"])
        self.assertTrue(any("Stokes limit only" in p for p in case_problems(inertial, stokes_only=True)))


class ProvenanceTests(unittest.TestCase):
    def test_commit_file_of_an_archived_source_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "COMMIT").write_text("abc1234\n")
            self.assertEqual(component_commit(Path(tmp)), "abc1234")

    def test_source_without_commit_or_git_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(component_commit(Path(tmp)), "unknown")

    def test_pyoomph_revision_is_the_installed_commit(self) -> None:
        if importlib.util.find_spec("pyoomph") is None:
            self.skipTest("needs the pinned pyoomph environment")
        info = pyoomph_revision()
        self.assertTrue(info["version"])
        pin = (ROOT / "pyproject.toml").read_text()
        if info["commit"] is not None:
            self.assertIn(info["commit"], pin)


class BimRunnerRefusalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bim = load(ROOT / "verificationCases" / "bim-stokes" / "run_bim_stokes.py")

    def run_bim(self, out: Path, *extra: str) -> str:
        argv = ["run_bim_stokes.py", str(STOKES_CASE), "--out", str(out), *extra]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stop:
            self.bim.main()
        return str(stop.exception)

    def test_fresh_run_into_an_existing_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "neck.csv").write_text("t,R_min\n0,1e-6\n")
            self.assertIn("already holds a run", self.run_bim(out))
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["neck.csv"])

    def test_checkpoint_of_another_case_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent"
            (parent / "restart").mkdir(parents=True)
            checkpoint = parent / "restart" / "state_000025.npz"
            np.savez(checkpoint, t=0.0)
            out = Path(tmp) / "continuation"
            self.assertIn("cannot be checked", self.run_bim(out, "--restart-from", str(checkpoint)))
            (parent / "run-manifest.json").write_text(json.dumps(
                {"solver": "coalescence.bim", "case_sha256": "0" * 64, "case_id": "other"}))
            self.assertIn("another case", self.run_bim(out, "--restart-from", str(checkpoint)))
            self.assertFalse(out.exists())


class SeedGateTests(unittest.TestCase):
    def test_non_finite_residuals_never_pass(self) -> None:
        from coalescence.fem.stokes_block_audit import residual_fell
        self.assertTrue(residual_fell(1.0, 1e-7))
        self.assertTrue(residual_fell(1e3, 1e-4))
        self.assertFalse(residual_fell(0.5, 2e-6))
        for before, after in ((1.0, math.nan), (math.nan, 1e-9), (1.0, math.inf), (math.inf, 1.0)):
            self.assertFalse(residual_fell(before, after), (before, after))


class DropFitTests(unittest.TestCase):
    def test_energy_budget_uses_exact_period_boundaries(self) -> None:
        fit = load(ROOT / "verificationCases" / "drop-oscillation" / "fit_drop_oscillation.py")
        t = np.linspace(0.0, 3.05, 7001)              # samples do not fall on whole periods
        energy, dissipated = -t, t ** 2               # budget over [a, b] is 1 - 1/(a + b)
        result = fit.energy_budget_per_period(t, energy, dissipated, period=1.0)
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result, [1 - 1 / 3, 1 - 1 / 5], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
