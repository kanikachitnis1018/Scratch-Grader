import unittest

from src.few_shot_prompt import build_few_shot_prompt, build_prompt_chain_of_thought


class FewShotPromptTests(unittest.TestCase):
    def test_build_few_shot_prompt_includes_examples_and_target(self):
        records = [
            {
                "id": 1,
                "features": {
                    "sprite_count": 3,
                    "block_count": 10,
                    "variable_count": 1,
                    "list_count": 0,
                    "uses_loops": True,
                    "uses_conditionals": False,
                    "uses_variables": True,
                    "uses_event_handling": True,
                    "uses_cloning": False,
                    "uses_collision": False,
                    "uses_animation": False,
                    "uses_sound": False,
                    "uses_ui_feedback": False,
                    "uses_lists": False,
                    "uses_math": False,
                    "uses_messaging": False,
                    "uses_algorithms": False,
                    "uses_nesting": False,
                    "uses_integration": False,
                },
                "grades": {
                    "problem_decomposition": 1,
                    "sequencing": 1,
                    "loops": 1,
                    "conditionals": 0,
                    "variables": 1,
                    "event_handling": 1,
                    "debugging": 0,
                    "procedures": 0,
                    "coordinates": 0,
                    "cloning": 0,
                    "collision": 0,
                    "animation": 0,
                    "sound": 0,
                    "ui_feedback": 0,
                    "lists": 0,
                    "math": 0,
                    "messaging": 0,
                    "algorithms": 0,
                    "nesting": 0,
                    "integration": 0,
                },
            },
            {
                "id": 2,
                "features": {
                    "sprite_count": 8,
                    "block_count": 200,
                    "variable_count": 5,
                    "list_count": 2,
                    "uses_loops": True,
                    "uses_conditionals": True,
                    "uses_variables": True,
                    "uses_event_handling": True,
                    "uses_cloning": True,
                    "uses_collision": True,
                    "uses_animation": True,
                    "uses_sound": True,
                    "uses_ui_feedback": True,
                    "uses_lists": True,
                    "uses_math": True,
                    "uses_messaging": True,
                    "uses_algorithms": True,
                    "uses_nesting": True,
                    "uses_integration": True,
                },
                "grades": {
                    "problem_decomposition": 4,
                    "sequencing": 5,
                    "loops": 5,
                    "conditionals": 5,
                    "variables": 4,
                    "event_handling": 2,
                    "debugging": 0,
                    "procedures": 3,
                    "coordinates": 3,
                    "cloning": 3,
                    "collision": 1,
                    "animation": 1,
                    "sound": 5,
                    "ui_feedback": 3,
                    "lists": 4,
                    "math": 2,
                    "messaging": 2,
                    "algorithms": 4,
                    "nesting": 4,
                    "integration": 0,
                },
            },
        ]

        prompt = build_few_shot_prompt(records, num_examples=1)

        self.assertIn("Example 1:", prompt)
        self.assertIn("Features:", prompt)
        self.assertIn("Grades:", prompt)
        self.assertIn("Now grade this new project:", prompt)
        self.assertIn("sprite_count: 8", prompt)

    def test_build_prompt_chain_of_thought_structure(self):
        records = [
            {
                "id": 1,
                "features": {
                    "sprite_count": 3,
                    "block_count": 10,
                    "variable_count": 1,
                    "list_count": 0,
                    "uses_loops": True,
                    "uses_conditionals": False,
                    "uses_variables": True,
                    "uses_event_handling": True,
                    "uses_cloning": False,
                    "uses_collision": False,
                    "uses_animation": False,
                    "uses_sound": False,
                    "uses_ui_feedback": False,
                    "uses_lists": False,
                    "uses_math": False,
                    "uses_messaging": False,
                    "uses_algorithms": False,
                    "uses_nesting": False,
                    "uses_integration": False,
                },
                "grades": {
                    "problem_decomposition": 1, "sequencing": 1, "loops": 1,
                    "conditionals": 0, "variables": 1, "event_handling": 1,
                    "debugging": 0, "procedures": 0, "coordinates": 0,
                    "cloning": 0, "collision": 0, "animation": 0, "sound": 0,
                    "ui_feedback": 0, "lists": 0, "math": 0, "messaging": 0,
                    "algorithms": 0, "nesting": 0, "integration": 0,
                },
            },
            {
                "id": 2,
                "features": {
                    "sprite_count": 8,
                    "block_count": 200,
                    "variable_count": 5,
                    "list_count": 2,
                    "uses_loops": True,
                    "uses_conditionals": True,
                    "uses_variables": True,
                    "uses_event_handling": True,
                    "uses_cloning": True,
                    "uses_collision": True,
                    "uses_animation": True,
                    "uses_sound": True,
                    "uses_ui_feedback": True,
                    "uses_lists": True,
                    "uses_math": True,
                    "uses_messaging": True,
                    "uses_algorithms": True,
                    "uses_nesting": True,
                    "uses_integration": True,
                },
                "grades": {
                    "problem_decomposition": 4, "sequencing": 5, "loops": 5,
                    "conditionals": 5, "variables": 4, "event_handling": 2,
                    "debugging": 0, "procedures": 3, "coordinates": 3,
                    "cloning": 3, "collision": 1, "animation": 1, "sound": 5,
                    "ui_feedback": 3, "lists": 4, "math": 2, "messaging": 2,
                    "algorithms": 4, "nesting": 4, "integration": 0,
                },
            },
        ]

        prompt = build_prompt_chain_of_thought(records, num_examples=1)

        # Existing few-shot structure must be preserved
        self.assertIn("Example 1:", prompt)
        self.assertIn("Features:", prompt)
        self.assertIn("Grades:", prompt)
        self.assertIn("Now grade this new project:", prompt)

        # Reasoning section must be present
        self.assertIn("Reasoning:", prompt)

        # Features must still appear for both example and target
        self.assertIn("sprite_count: 3", prompt)
        self.assertIn("sprite_count: 8", prompt)

        # Grades must still appear in example
        self.assertIn("problem_decomposition: 1", prompt)

        # Final instruction must ask for JSON only
        self.assertIn("ONLY the final grades", prompt)
        self.assertIn("strictly valid JSON", prompt)


if __name__ == "__main__":
    unittest.main()
