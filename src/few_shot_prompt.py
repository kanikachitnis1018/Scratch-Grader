import json
from typing import List, Dict, Any


def _format_features(features: Dict[str, Any]) -> str:
    lines = []
    for key, value in features.items():
        if isinstance(value, bool):
            rendered = "True" if value else "False"
        elif isinstance(value, str):
            rendered = value
        else:
            rendered = str(value)
        lines.append(f"  {key}: {rendered}")
    return "\n".join(lines)


def _format_grades(grades: Dict[str, Any]) -> str:
    lines = [f"  {key}: {value}" for key, value in grades.items()]
    return "\n".join(lines)


def build_few_shot_prompt(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    blocks = []
    for index, record in enumerate(examples, start=1):
        block = "\n".join([
            f"Example {index}:",
            "Features:",
            _format_features(record["features"]),
            "Grades:",
            _format_grades(record["grades"]),
        ])
        blocks.append(block)

    target_block = "\n".join([
        "Now grade this new project:",
        "Features:",
        _format_features(target["features"]),
    ])
    blocks.append(target_block)
    blocks.append("Return only the grades as a JSON object with the same rubric keys.")
    return "\n\n".join(blocks)


def _generate_reasoning(features: Dict[str, Any], grades: Dict[str, Any]) -> str:
    """Generate a step-by-step reasoning section that links features to rubric scores."""
    steps = []

    sprite_count = features.get("sprite_count", 0)
    block_count = features.get("block_count", 0)
    variable_count = features.get("variable_count", 0)
    list_count = features.get("list_count", 0)

    steps.append(
        f"  - The project has {sprite_count} sprite(s) and {block_count} block(s), "
        f"giving a rough sense of overall complexity."
    )

    dimension_hints = {
        "problem_decomposition": (
            f"  - problem_decomposition ({grades.get('problem_decomposition', '?')}): "
            f"{sprite_count} sprite(s) and {block_count} block(s) suggest "
            + ("strong decomposition across multiple actors."
               if sprite_count >= 5 else "limited decomposition.")
        ),
        "sequencing": (
            f"  - sequencing ({grades.get('sequencing', '?')}): "
            f"{block_count} block(s) indicate "
            + ("a well-ordered sequence of instructions."
               if block_count >= 50 else "a short or simple sequence.")
        ),
        "loops": (
            f"  - loops ({grades.get('loops', '?')}): "
            + ("Loop blocks detected — repetition is present."
               if features.get("uses_loops") else "No loop blocks found.")
        ),
        "conditionals": (
            f"  - conditionals ({grades.get('conditionals', '?')}): "
            + ("Conditional blocks detected — branching logic is present."
               if features.get("uses_conditionals") else "No conditional blocks found.")
        ),
        "variables": (
            f"  - variables ({grades.get('variables', '?')}): "
            f"{variable_count} variable(s) defined. "
            + ("Variables are actively used."
               if features.get("uses_variables") else "No variable usage detected.")
        ),
        "event_handling": (
            f"  - event_handling ({grades.get('event_handling', '?')}): "
            + ("Event blocks detected — project responds to triggers."
               if features.get("uses_event_handling") else "No event blocks detected.")
        ),
        "debugging": (
            f"  - debugging ({grades.get('debugging', '?')}): "
            "Debugging evidence is not directly detectable from block counts alone."
        ),
        "procedures": (
            f"  - procedures ({grades.get('procedures', '?')}): "
            + ("Custom procedure blocks detected — code is modularised."
               if features.get("uses_algorithms") and block_count >= 30
               else "No custom procedure blocks detected.")
        ),
        "coordinates": (
            f"  - coordinates ({grades.get('coordinates', '?')}): "
            + ("Coordinate-based motion blocks detected."
               if features.get("uses_event_handling") and sprite_count >= 2
               else "No explicit coordinate usage detected.")
        ),
        "cloning": (
            f"  - cloning ({grades.get('cloning', '?')}): "
            + ("Clone blocks detected — dynamic sprite creation is used."
               if features.get("uses_cloning") else "No cloning detected.")
        ),
        "collision": (
            f"  - collision ({grades.get('collision', '?')}): "
            + ("Collision/touching sensing blocks detected."
               if features.get("uses_collision") else "No collision detection found.")
        ),
        "animation": (
            f"  - animation ({grades.get('animation', '?')}): "
            + ("Costume-switching blocks detected — animation is present."
               if features.get("uses_animation") else "No animation blocks detected.")
        ),
        "sound": (
            f"  - sound ({grades.get('sound', '?')}): "
            + ("Sound blocks detected — audio is used."
               if features.get("uses_sound") else "No sound blocks detected.")
        ),
        "ui_feedback": (
            f"  - ui_feedback ({grades.get('ui_feedback', '?')}): "
            + ("UI feedback blocks (say/think/visuals) detected."
               if features.get("uses_ui_feedback") else "No UI feedback blocks detected.")
        ),
        "lists": (
            f"  - lists ({grades.get('lists', '?')}): "
            f"{list_count} list(s) defined. "
            + ("Lists are used for data storage."
               if features.get("uses_lists") else "No list usage detected.")
        ),
        "math": (
            f"  - math ({grades.get('math', '?')}): "
            + ("Math/operator blocks detected."
               if features.get("uses_math") else "No math operator blocks detected.")
        ),
        "messaging": (
            f"  - messaging ({grades.get('messaging', '?')}): "
            + ("Broadcast/message blocks detected — inter-sprite communication is used."
               if features.get("uses_messaging") else "No broadcast blocks detected.")
        ),
        "algorithms": (
            f"  - algorithms ({grades.get('algorithms', '?')}): "
            + (f"{block_count} block(s) with loops and conditionals suggest algorithmic thinking."
               if features.get("uses_loops") and features.get("uses_conditionals")
               else "Limited algorithmic structure detected.")
        ),
        "nesting": (
            f"  - nesting ({grades.get('nesting', '?')}): "
            + ("Loops and conditionals both present — nesting is likely."
               if features.get("uses_nesting") else "No evidence of nested control structures.")
        ),
        "integration": (
            f"  - integration ({grades.get('integration', '?')}): "
            + ("Multiple concepts combined across sprites — integration is evident."
               if features.get("uses_integration") else "Limited cross-concept integration detected.")
        ),
    }

    for step in dimension_hints.values():
        steps.append(step)

    return "\n".join(steps)


def build_prompt_chain_of_thought(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """Build a few-shot prompt with step-by-step reasoning for each example.

    The existing examples and grades are preserved exactly. A Reasoning section
    is inserted between Features and Grades for every example to guide the model
    through how the features connect to each rubric dimension.
    """
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    blocks = []
    for index, record in enumerate(examples, start=1):
        reasoning = _generate_reasoning(record["features"], record["grades"])
        block = "\n".join([
            f"Example {index}:",
            "Features:",
            _format_features(record["features"]),
            "Reasoning:",
            reasoning,
            "Grades:",
            _format_grades(record["grades"]),
        ])
        blocks.append(block)

    target_block = "\n".join([
        "Now grade this new project:",
        "Features:",
        _format_features(target["features"]),
        "",
        "Think step-by-step about how the features relate to each rubric dimension.",
        "Then output ONLY the final grades as a strictly valid JSON object with the same rubric keys.",
    ])
    blocks.append(target_block)
    return "\n\n".join(blocks)


def save_prompt_to_file(records: List[Dict[str, Any]], output_path: str, num_examples: int = 3) -> str:
    prompt = build_few_shot_prompt(records, num_examples=num_examples)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    return prompt
