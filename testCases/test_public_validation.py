"""The public figures retain ratio direction, units and bounded interpolation."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "postProcess"))
from public_validation_data import at_radius, export_inputs, find_run, load_inputs, stats
from plot_public_validation import fem_over_markers, read_markers
from plot_finite_oh_validation import comparison_metrics


class PublicValidationTests(unittest.TestCase):
    def test_fem_over_anthony_direction(self):
        data = {"R_min": np.array([1., 10.]), "u_neck": np.array([4., 2.])}
        radius, error = fem_over_markers(np.array([[1., 2.], [10., 4.], [100., 9.]]), data)
        np.testing.assert_array_equal(radius, [1., 10.])
        np.testing.assert_allclose(error, [100., -50.])

    def test_no_radius_extrapolation(self):
        data = {"R_min": np.array([1., 10.]), "u_neck": np.array([4., 2.])}
        self.assertAlmostEqual(float(at_radius(data, np.sqrt(10))), 3)
        with self.assertRaisesRegex(ValueError, "extrapolate"):
            at_radius(data, [0.9, 1.])

    def test_missing_run_names_identity(self):
        with self.assertRaisesRegex(ValueError, "missing-run"):
            find_run([], "missing-run")

    def test_empty_marker_series_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "markers.csv"
            with path.open("w", newline="") as file:
                csv.writer(file).writerow(["series", "R_min", "u_v"])
            with self.assertRaisesRegex(ValueError, "no markers.*missing-series"):
                read_markers(path, "missing-series")

    def test_bundle_roundtrip_and_checksum(self):
        histories = {"finite": {"R_min": np.array([1., 2.])}}
        metadata = {"schema_version": 1, "histories": {"finite": {"columns": ["R_min"]}}}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            export_inputs(directory, histories, metadata)
            actual, _ = load_inputs(directory)
            np.testing.assert_array_equal(actual["finite"]["R_min"], [1., 2.])
            with self.assertRaises(FileExistsError):
                export_inputs(directory, histories, metadata)
            with (directory / "curves.npz").open("ab") as file:
                file.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_inputs(directory)

    def test_empty_statistics_rejected(self):
        with self.assertRaisesRegex(ValueError, "no comparison samples"):
            stats([])

    def test_public_bundle_abscissae_and_deficit(self):
        histories, metadata = load_inputs()
        fits, _, comparison = comparison_metrics(histories)
        self.assertEqual(comparison["radius"]["abscissa"], "tau_v")
        self.assertEqual(comparison["velocity"]["abscissa"], "R_min")
        self.assertEqual(metadata["units"], "visco-capillary")
        self.assertAlmostEqual(fits["finite"].t_con, 1.5201673587e-7, delta=1e-15)
        deficit = 100 * (at_radius(histories["finite"], 0.03) /
                         at_radius(histories["stokes"], 0.03) - 1)
        self.assertAlmostEqual(float(deficit), -12.15056461, places=6)


if __name__ == "__main__":
    unittest.main()
