"""Contact-time fitting and scaling contracts for the Fig. 3 postprocessor."""

from __future__ import annotations

import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from postprocess.fig3 import fit_contact_time, fig3_series, read_progress_neck


class Fig3ContactTimeTests(unittest.TestCase):
    def test_recovers_synthetic_power_law_contact_time(self) -> None:
        expected_shift = 0.25
        expected_exponent = 0.63
        prefactor = 2.5e-4
        time = np.linspace(0.0, 100.0, 81)
        radius = prefactor * (time + expected_shift) ** expected_exponent

        fit = fit_contact_time(time, radius)

        self.assertEqual(fit.first_radius, radius[0])
        self.assertGreaterEqual(fit.last_radius, 10.0 * radius[0])
        self.assertGreaterEqual(fit.point_count, 4)
        self.assertAlmostEqual(fit.t_con, expected_shift, delta=2.0e-6)
        self.assertAlmostEqual(fit.exponent, expected_exponent, delta=2.0e-7)
        self.assertAlmostEqual(fit.prefactor, prefactor, delta=2.0e-10)
        self.assertLess(fit.log_rms, 1.0e-9)

    def test_refuses_contact_time_before_a_decade_of_growth(self) -> None:
        time = np.linspace(0.0, 1.0, 20)
        radius = 1.0e-6 * (1.0 + 3.0 * time)
        with self.assertRaisesRegex(ValueError, "one decade"):
            fit_contact_time(time, radius)

    def test_rejects_non_monotone_time(self) -> None:
        time = np.array([0.0, 1.0, 0.5, 2.0, 3.0])
        radius = np.array([1.0, 10.0, 11.0, 12.0, 13.0])
        with self.assertRaisesRegex(ValueError, "strictly"):
            fit_contact_time(time, radius)

    def test_fig3_rescaling_uses_fitted_shift(self) -> None:
        expected_shift = 0.1
        time = np.linspace(0.0, 20.0, 81)
        radius = 1.0e-4 * (time + expected_shift) ** 0.8
        velocity = 0.2 + 0.01 * time
        data = fig3_series(
            {"t": time, "R_min": radius, "u_min": velocity},
            inertia=True,
            oh=0.6,
        )
        self.assertAlmostEqual(data["t_con"][0], expected_shift, delta=2.0e-6)
        np.testing.assert_allclose(data["tau_v"], (time + expected_shift) / 0.6)
        np.testing.assert_allclose(data["u_v"], velocity * 0.6)

    def test_post_decade_output_density_does_not_change_fit(self) -> None:
        shift = 0.2
        exponent = 0.7
        early_time = np.linspace(0.0, 12.0, 25)
        sparse_time = np.concatenate((early_time, np.linspace(20.0, 40.0, 4)))
        dense_time = np.concatenate((early_time, np.linspace(12.1, 40.0, 200)))
        sparse_radius = 1.0e-4 * (sparse_time + shift) ** exponent
        dense_radius = 1.0e-4 * (dense_time + shift) ** exponent

        sparse = fit_contact_time(sparse_time, sparse_radius)
        dense = fit_contact_time(dense_time, dense_radius)

        self.assertAlmostEqual(sparse.t_con, dense.t_con, delta=1.0e-10)
        self.assertAlmostEqual(sparse.exponent, dense.exponent, delta=1.0e-10)

    def test_progress_restart_record_wins_at_overlapping_time(self) -> None:
        with tempfile.TemporaryDirectory(prefix="fig3-progress-") as directory:
            attempt = Path(directory) / "runs" / "attempts" / "attempt-0001"
            path = attempt / "runtime" / "progress.jsonl"
            path.parent.mkdir(parents=True)
            (attempt / "base").mkdir()
            checkpoint = attempt / "base" / "checkpoint.state"
            checkpoint.write_bytes(b"state")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for segment in ("restart-0001", "abandoned-sibling"):
                owner = attempt / "segments" / segment
                owner.mkdir(parents=True)
                (owner / "restart.json").write_text(json.dumps({
                    "schema": "pyoomph-restart-segment-v1",
                    "segment_id": segment,
                    "parent": "base",
                    "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest},
                }))
            records = [
                {"event": "initial-state", "segment_id": "base", "physical_time": 0.0,
                 "R_min": 1.0e-6, "u_min": 0.0, "Z_b": 5.0e-13,
                 "abs_2H": 2.0e12, "V_bod": 0.0},
                {"event": "accepted-step", "segment_id": "base", "physical_time": 1.0,
                 "R_min": 1.0e-5, "u_min": 2.0, "Z_b": 3.0e-10,
                 "abs_2H": 4.0e12, "V_bod": 5.0},
                {"event": "accepted-step", "segment_id": "restart-0001", "physical_time": 1.0,
                 "R_min": 1.1e-5, "u_min": 2.1, "Z_b": 3.1e-10,
                 "abs_2H": 4.1e12, "V_bod": 5.1},
                {"event": "accepted-step", "segment_id": "restart-0001", "physical_time": 2.0,
                 "R_min": 2.0e-5, "u_min": 3.0, "Z_b": 4.0e-10,
                 "abs_2H": 5.0e12, "V_bod": 6.0},
                {"event": "accepted-step", "segment_id": "abandoned-sibling", "physical_time": 2.0,
                 "R_min": 9.0e-5, "u_min": 9.0, "Z_b": 9.0e-10,
                 "abs_2H": 9.0e12, "V_bod": 9.0},
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records))
            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"event":')

            with self.assertRaisesRegex(ValueError, "multiple restart branches"):
                read_progress_neck(path)
            data = read_progress_neck(path, terminal_segment="restart-0001")

            np.testing.assert_allclose(data["t"], [0.0, 1.0, 2.0])
            np.testing.assert_allclose(data["R_min"], [1.0e-6, 1.1e-5, 2.0e-5])
            np.testing.assert_allclose(data["u_min"], [0.0, 2.1, 3.0])


if __name__ == "__main__":
    unittest.main()
