"""Runner and figure-script contracts that need no solver run.

A runner exits 0 only when the run reached its stop radius; a restart is refused
in a directory that already holds a run (restart segments are separate runs joined
by ``continues``); the startup-family script refuses an endpoint outside a run.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "validationCases" / "anthony2020" / "T5-stokes-R0-1e-06.json"


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bim = load(ROOT / "verificationCases" / "bim-stokes" / "run_bim_stokes.py")
        cls.fem = load(ROOT / "simulationCases" / "run_coalescence.py")

    def test_exit_status_is_zero_only_at_the_stop_radius(self) -> None:
        for runner in (self.bim, self.fem):
            self.assertEqual(runner.exit_status({"status": "reached_R_stop"}), 0)
            for status in ("wall_limit", "limit", "newton_failure", "running"):
                self.assertEqual(runner.exit_status({"status": status}), 1, (runner.__name__, status))
        # Only the FEM runner has a stop time.
        self.assertEqual(self.fem.exit_status({"status": "reached_t_stop"}), 0)
        self.assertEqual(self.bim.exit_status({"status": "reached_t_stop"}), 1)

    def test_restart_into_an_existing_run_is_refused(self) -> None:
        for runner, name in ((self.bim, "run_bim_stokes.py"), (self.fem, "run_coalescence.py")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                (out / "neck.csv").write_text("t,R_min,u_neck,tip_radius\n")
                argv = [name, str(CASE), "--out", str(out), "--restart-from", str(out / "missing.npz")]
                if runner is self.fem:
                    argv.append("--seed-frozen-stokes")
                with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stop:
                    runner.main()
                self.assertIn("own output directory", str(stop.exception))
                self.assertEqual(sorted(p.name for p in out.iterdir()), ["neck.csv"])


class FigureScriptTests(unittest.TestCase):
    def test_family_endpoint_outside_a_run_is_refused(self) -> None:
        family = load(ROOT / "postProcess" / "plot_startup_family.py")
        R = np.array([1e-6, 1e-4, 0.0303])
        family.check_endpoint("run", R, 0.03)
        for endpoint in (1e-7, 1e-6, 0.05):
            with self.assertRaises(SystemExit):
                family.check_endpoint("run", R, endpoint)


class Fig3ScriptTests(unittest.TestCase):
    def test_relative_refuses_a_non_increasing_abscissa(self) -> None:
        fig3 = load(ROOT / "validationCases" / "anthony2020" / "plot_anthony_fig3.py")
        ref = np.array([[2e-6, 1.0], [3e-6, 1.0]])
        fig3.relative(ref, np.array([1e-6, 2e-6, 4e-6]), np.array([1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            fig3.relative(ref, np.array([1e-6, 4e-6, 2e-6]), np.array([1.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
