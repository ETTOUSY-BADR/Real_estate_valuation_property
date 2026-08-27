"""Leakage-safe national preflight and clustering checks."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paris_avm.analysis.phase4_preflight import CLUSTER_INPUTS, fit_department_clusters


class Phase4PreflightTests(unittest.TestCase):
    def test_department_clusters_exclude_current_transaction_target(self) -> None:
        self.assertNotIn("valeur_fonciere", CLUSTER_INPUTS)
        self.assertNotIn("price_per_m2", CLUSTER_INPUTS)

    def test_cluster_fit_is_deterministic_and_covers_departments(self) -> None:
        rows: list[dict[str, object]] = []
        for department in range(1, 9):
            for index in range(20):
                rows.append(
                    {
                        "code_departement": f"{department:02d}",
                        "code_region": "84" if department <= 4 else "11",
                        "surface_reelle_bati": 35 + department * 5 + index / 10,
                        "rooms_per_10m2": 0.4 + department / 100,
                        "weighted_comp_ppm2": (
                            np.nan if index == 0 else 2_000 + department * 800 + index
                        ),
                        "local_365d_500m_count": department * 3 + index,
                        "effective_comparable_count": department + index / 4,
                        "latitude": 42 + department / 2,
                        "longitude": -2 + department / 2,
                    }
                )
        data = pd.DataFrame(rows)
        first, first_meta = fit_department_clusters(data, requested_clusters=3)
        second, second_meta = fit_department_clusters(data, requested_clusters=3)

        self.assertEqual(first["market_cluster"].tolist(), second["market_cluster"].tolist())
        self.assertEqual(first["code_departement"].nunique(), 8)
        self.assertEqual(first_meta["selected_clusters"], 3)
        self.assertFalse(first_meta["target_used"])
        self.assertEqual(first_meta["silhouette_scores"], second_meta["silhouette_scores"])


if __name__ == "__main__":
    unittest.main()
