import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.generate_prompt_from_test_json import write_prompt_for_url


class PromptGenerationTests(unittest.TestCase):
    def test_write_prompt_for_url_overwrites_existing_prompt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "few_shot_prompt.txt"
            output_path.write_text("old prompt", encoding="utf-8")

            records = [
                {
                    "id": 1,
                    "features": {"sprite_count": 1, "block_count": 2, "uses_loops": False, "uses_conditionals": False},
                    "grades": {"problem_decomposition": 1, "sequencing": 1},
                }
            ]

            with patch("src.generate_prompt_from_test_json.load_dataset_records", return_value=records), \
                 patch("src.generate_prompt_from_test_json.get_project_features_from_url", return_value={"sprite_count": 5, "block_count": 10, "uses_loops": True, "uses_conditionals": True}):
                prompt = write_prompt_for_url("https://scratch.mit.edu/projects/123", output_path=output_path)

            self.assertIn("Now grade this new project:", prompt)
            self.assertNotIn("old prompt", output_path.read_text(encoding="utf-8"))
            self.assertIn("sprite_count=5", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
