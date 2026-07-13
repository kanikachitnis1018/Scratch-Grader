import unittest

from src.loocv import run_loocv


class LoocvTests(unittest.TestCase):
    def test_run_loocv_returns_summary_metrics(self):
        records = []
        for idx in range(3):
            grades = {
                "problem_decomposition": 1,
                "sequencing": 1,
                "loops": 1,
                "conditionals": 1,
                "variables": 1,
                "event_handling": 1,
                "debugging": 1,
                "procedures": 1,
                "coordinates": 1,
                "cloning": 1,
                "collision": 1,
                "animation": 1,
                "sound": 1,
                "ui_feedback": 1,
                "lists": 1,
                "math": 1,
                "messaging": 1,
                "algorithms": 1,
                "nesting": 1,
                "integration": 1,
            }
            records.append({
                "id": idx + 1,
                "features": {"sprite_count": idx + 1},
                "grades": grades,
            })

        def fake_model(prompt):
            return {
                "problem_decomposition": 1,
                "sequencing": 1,
                "loops": 1,
                "conditionals": 1,
                "variables": 1,
                "event_handling": 1,
                "debugging": 1,
                "procedures": 1,
                "coordinates": 1,
                "cloning": 1,
                "collision": 1,
                "animation": 1,
                "sound": 1,
                "ui_feedback": 1,
                "lists": 1,
                "math": 1,
                "messaging": 1,
                "algorithms": 1,
                "nesting": 1,
                "integration": 1,
            }

        result = run_loocv(records, model_fn=fake_model)

        self.assertEqual(result["summary"]["num_samples"], 3)
        self.assertEqual(result["summary"]["num_dimensions"], 20)
        self.assertEqual(result["summary"]["accuracy"], 100.0)
        self.assertEqual(result["summary"]["cohens_kappa"], 1.0)
        self.assertTrue(all(value == 0.0 for value in result["summary"]["mae_by_dimension"].values()))
        # With only class 1 present in 0-5 range, macro averages are 1.0/6 ≈ 0.1667
        self.assertAlmostEqual(result["summary"]["macro_precision"], 1.0 / 6.0, places=3)
        self.assertAlmostEqual(result["summary"]["macro_recall"], 1.0 / 6.0, places=3)
        self.assertAlmostEqual(result["summary"]["macro_f1"], 1.0 / 6.0, places=3)
        self.assertEqual(len(result["sample_results"]), 3)

    def test_run_loocv_targets_the_held_out_sample(self):
        records = []
        for idx in range(5):
            records.append({
                "id": idx + 1,
                "features": {"sprite_count": idx + 1},
                "grades": {
                    "problem_decomposition": 1,
                    "sequencing": 1,
                    "loops": 1,
                    "conditionals": 1,
                    "variables": 1,
                    "event_handling": 1,
                    "debugging": 1,
                    "procedures": 1,
                    "coordinates": 1,
                    "cloning": 1,
                    "collision": 1,
                    "animation": 1,
                    "sound": 1,
                    "ui_feedback": 1,
                    "lists": 1,
                    "math": 1,
                    "messaging": 1,
                    "algorithms": 1,
                    "nesting": 1,
                    "integration": 1,
                },
            })

        prompts = []

        def fake_model(prompt):
            prompts.append(prompt)
            return {
                "problem_decomposition": 1,
                "sequencing": 1,
                "loops": 1,
                "conditionals": 1,
                "variables": 1,
                "event_handling": 1,
                "debugging": 1,
                "procedures": 1,
                "coordinates": 1,
                "cloning": 1,
                "collision": 1,
                "animation": 1,
                "sound": 1,
                "ui_feedback": 1,
                "lists": 1,
                "math": 1,
                "messaging": 1,
                "algorithms": 1,
                "nesting": 1,
                "integration": 1,
            }

        run_loocv(records, model_fn=fake_model)

        self.assertTrue(prompts)
        self.assertIn("Now grade this new project:", prompts[0])
        self.assertIn("sprite_count: 1", prompts[0])


if __name__ == "__main__":
    unittest.main()
