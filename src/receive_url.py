import os
import sys
from pathlib import Path

from generate_prompt_from_test_json import write_prompt_for_url


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/receive_url.py <scratch_url>")
        sys.exit(1)

    url = sys.argv[1]
    output_path = Path(__file__).resolve().parent.parent / "few_shot_prompt.txt"
    prompt = write_prompt_for_url(url, output_path=output_path)
    print(f"Processed URL: {url}")
    print(f"Prompt written to: {output_path}")
    print(prompt)


if __name__ == "__main__":
    main()
