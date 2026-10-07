"""The case-video and energy-budget tools refuse unusable inputs and write resolvable playback links."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postProcess"))
import reconstruct_stokes_fields
from check_energy_budget import budget
from render_case_dashboard_video import initial_radius
from render_hybrid_video import check_playback, link_sequence, playback_sequence


class PlaybackTests(unittest.TestCase):
    def test_links_resolve_with_relative_output(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)
        frames = Path("media") / "case-frames"
        frames.mkdir(parents=True)
        for k in range(2):
            (frames / f"frame-{k:05d}.png").write_bytes(b"png")
        seq_dir = Path("media") / "case-sequence"
        link_sequence(np.array([0, 0, 1]), frames, seq_dir)
        links = sorted(seq_dir.glob("seq-*.png"))
        self.assertEqual(len(links), 3)
        self.assertTrue(all(link.is_symlink() and link.exists() for link in links))
        self.assertEqual(links[2].resolve(), (frames / "frame-00001.png").resolve())

    def test_unplayable_inputs_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "two frames"):
            check_playback(np.array([0.0]), 32.0, 0.6, 30)
        for duration in (0.5, 0.6, 0.61):
            with self.subTest(duration=duration), self.assertRaisesRegex(SystemExit, "hold-first"):
                check_playback(np.array([0.0, 1e-3]), duration, 0.6, 30)
        with self.assertRaisesRegex(SystemExit, "negative"):
            check_playback(np.array([0.0, 1e-3]), 32.0, -0.1, 30)
        times = np.array([0.0, 1e-3, 1e-2])
        check_playback(times, 2.0, 0.6, 30)
        self.assertEqual(len(playback_sequence(times, 0.05, 2.0, 30, 0.6)), 60)


class ContactTimeRadiusTests(unittest.TestCase):
    def test_radius_is_the_initial_bridge_row(self):
        self.assertEqual(initial_radius({"t": np.array([0.0, 1e-3]), "R_min": np.array([1e-6, 2e-6])}), 1e-6)

    def test_history_without_initial_row_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "t = 0"):
            initial_radius({"t": np.array([1e-3, 2e-3]), "R_min": np.array([2e-6, 3e-6])})


class EnergyBudgetTests(unittest.TestCase):
    COLUMNS = ("t", "R_min", "kinetic", "area", "dissipation", "n_remesh")

    def runtime(self, rows: list[tuple[float, ...]]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with (Path(tmp.name) / "neck.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.COLUMNS)
            writer.writerows(rows)
        return Path(tmp.name)

    def test_empty_history_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "no rows"):
            budget(self.runtime([]))

    def test_no_usable_interval_is_refused(self):
        start = (0.0, 1e-3, 0.0, 10.0, 2.0, 0)
        for rows in ([start],
                     [start, (1e-3, 1.1e-3, 0.0, 9.998, 2.0, 0)],
                     [start, (1e-3, 1.1e-3, 0.0, 9.998, 2.0, 0), (2e-3, 1.2e-3, 0.0, 9.996, 2.0, 1)]):
            with self.subTest(rows=len(rows)), self.assertRaisesRegex(SystemExit, "no usable energy interval"):
                budget(self.runtime(rows))

    def test_closed_budget_has_zero_residual(self):
        rows = [(k * 1e-3, 1e-3 * (1 + 0.1 * k), 0.0, 10.0 - 2.0 * k * 1e-3, 2.0, 0) for k in range(4)]
        result = budget(self.runtime(rows))
        self.assertEqual(result["steps_used"], 2)
        self.assertAlmostEqual(result["relative_residual_median"], 0.0, places=9)


class ReconstructionTests(unittest.TestCase):
    def test_snapshot_mode_without_snapshots_fails(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        runtime = Path(tmp.name) / "runtime"
        (runtime / "snapshots").mkdir(parents=True)
        argv = ["reconstruct_stokes_fields.py", str(runtime), "--out", str(Path(tmp.name) / "out"),
                "--states", "snapshots"]
        with mock.patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "no snapshots"):
            reconstruct_stokes_fields.main()
        self.assertFalse((Path(tmp.name) / "out").exists())


if __name__ == "__main__":
    unittest.main()
