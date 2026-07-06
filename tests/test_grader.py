import unittest
from unittest.mock import patch

import pandas as pd

from src.grader import enrich_dataset, extract_project_features


class ExtractProjectFeaturesTests(unittest.TestCase):
    def test_enrich_dataset_skips_unavailable_projects(self):
        df = pd.DataFrame([
            {"ID": 1, "Reviewer": 5, "1.Decomp": 5},
            {"ID": 2, "Reviewer": 4, "1.Decomp": 4},
        ])

        with patch("src.grader.fetch_project_json") as mock_fetch:
            mock_fetch.side_effect = [None, {"targets": []}]
            enriched = enrich_dataset(df)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["id"], 2)
        self.assertEqual(enriched[0]["grades"]["problem_decomposition"], 4)
        self.assertIn("features", enriched[0])

    def test_enrich_dataset_keeps_successful_empty_projects(self):
        df = pd.DataFrame([{"ID": 1, "Reviewer": 5}])

        with patch("src.grader.fetch_project_json", return_value={"targets": []}):
            enriched = enrich_dataset(df)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["id"], 1)
        self.assertEqual(enriched[0]["grades"]["problem_decomposition"], 0)
        self.assertIn("features", enriched[0])

    def test_enrich_dataset_uses_url_project_id_when_present(self):
        df = pd.DataFrame([{"ID": 815, "Reviewer": 5, "URL": "https://scratch.mit.edu/projects/1255366724"}])

        with patch("src.grader.fetch_project_json", return_value={"targets": []}) as mock_fetch:
            enriched = enrich_dataset(df)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["id"], 1255366724)
        mock_fetch.assert_called_once_with(1255366724)

    def test_enrich_dataset_converts_blank_rubric_scores_to_zero(self):
        df = pd.DataFrame([{"ID": 815, "Reviewer": 5, "URL": "https://scratch.mit.edu/projects/1255366724", "1.Decomp": None}])

        with patch("src.grader.fetch_project_json", return_value={"targets": []}):
            enriched = enrich_dataset(df)

        self.assertEqual(enriched[0]["grades"]["problem_decomposition"], 0)

    def test_extract_project_features_returns_compact_rubric_features(self):
        project_json = {
            "targets": [
                {
                    "isStage": True,
                    "variables": {},
                    "lists": {},
                    "blocks": {},
                    "sounds": [],
                    "costumes": [],
                },
                {
                    "isStage": False,
                    "name": "Sprite1",
                    "variables": {"score": ["score", 0]},
                    "lists": {},
                    "blocks": {
                        "a": {
                            "opcode": "event_whenflagclicked",
                            "next": "b",
                            "parent": None,
                            "inputs": {},
                            "fields": {},
                        },
                        "b": {
                            "opcode": "control_repeat",
                            "next": "c",
                            "parent": "a",
                            "inputs": {},
                            "fields": {},
                        },
                        "c": {
                            "opcode": "control_if",
                            "next": "d",
                            "parent": "b",
                            "inputs": {},
                            "fields": {},
                        },
                        "d": {
                            "opcode": "data_setvariableto",
                            "next": "e",
                            "parent": "c",
                            "inputs": {},
                            "fields": {},
                        },
                        "e": {
                            "opcode": "motion_gotoxy",
                            "next": "f",
                            "parent": "d",
                            "inputs": {},
                            "fields": {},
                        },
                        "f": {
                            "opcode": "event_broadcast",
                            "next": None,
                            "parent": "e",
                            "inputs": {},
                            "fields": {"BROADCAST_OPTION": ["message1"]},
                        },
                    },
                    "sounds": [{"name": "boop"}],
                    "costumes": [{"name": "costume1"}, {"name": "costume2"}],
                },
            ]
        }

        features = extract_project_features(project_json)

        self.assertEqual(features["sprite_count"], 1)
        self.assertEqual(features["block_count"], 6)
        self.assertEqual(features["variable_count"], 1)
        self.assertTrue(features["uses_loops"])
        self.assertTrue(features["uses_conditionals"])
        self.assertTrue(features["uses_variables"])
        self.assertTrue(features["uses_messaging"])
        self.assertTrue(features["uses_nesting"])
        self.assertTrue(features["uses_integration"])


if __name__ == "__main__":
    unittest.main()
