"""Run lists: checksum enforcement and joining a continuation to its parent."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coalescence.analysis.runs import load_runs, load_series, neck_file


def write_neck(path: Path, t, R, u) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["t,R_min,u_neck,tip_radius"] + [f"{a!r},{b!r},{c!r},1e-9" for a, b, c in zip(t, R, u)]
    path.write_text("\n".join(rows) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.roots = [root / "data"]
        base = write_neck(root / "data/base/runtime/neck.csv", [1.0, 2.0, 3.0, 4.0], [1.1, 1.2, 1.3, 1.4], [5.0, 5.1, 5.2, 5.3])
        cont = write_neck(root / "data/cont/runtime/neck.csv", [3.0, 5.0], [1.35, 1.5], [5.25, 5.4])
        self.listing = root / "runs.toml"
        self.listing.write_text(
            '[[run]]\nid = "base"\nsolver = "bim"\ncase = "c.json"\nR0 = 1.0\ncommit = "a"\nruntime = "runtime"\n'
            f'neck_sha256 = "{base}"\nseries = "linear"\n\n'
            '[[run]]\nid = "cont"\nsolver = "bim"\ncase = "c.json"\nR0 = 1.0\ncommit = "b"\nruntime = "runtime"\n'
            f'neck_sha256 = "{cont}"\nseries = "newton"\ncontinues = "base"\n')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_continuation_starts_from_the_parent_radius_at_restart(self) -> None:
        runs = load_runs(self.listing)
        newton = load_series(runs, runs[1], self.roots)
        np.testing.assert_allclose(newton["t"], [3.0, 5.0])
        np.testing.assert_allclose(newton["R_step"], [np.sqrt(1.2 * 1.35), np.sqrt(1.35 * 1.5)])
        linear = load_series(runs, runs[0], self.roots)
        np.testing.assert_allclose(linear["R_step"][0], np.sqrt(1.0 * 1.1))

    def test_checksum_mismatch_is_refused(self) -> None:
        runs = load_runs(self.listing)
        path = self.roots[0] / "base/runtime/neck.csv"
        path.write_text(path.read_text() + "5.0,1.5,5.4,1e-9\n")
        with self.assertRaises(ValueError):
            neck_file(runs[0], self.roots)


if __name__ == "__main__":
    unittest.main()
