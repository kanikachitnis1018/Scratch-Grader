import unittest

from src.server import build_response_payload


class ServerResponseTests(unittest.TestCase):
    def test_build_response_payload_excludes_prompt_content(self):
        payload = build_response_payload(
            url="https://scratch.mit.edu/projects/123",
            output_path="/tmp/few_shot_prompt.txt",
            grades={"problem_decomposition": 4},
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["url"], "https://scratch.mit.edu/projects/123")
        self.assertEqual(payload["prompt_path"], "/tmp/few_shot_prompt.txt")
        self.assertEqual(payload["grades"], {"problem_decomposition": 4})
        self.assertNotIn("prompt", payload)


if __name__ == "__main__":
    unittest.main()
