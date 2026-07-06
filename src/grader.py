import re

import pandas as pd

try:
    from .scratch_loader import fetch_project_json
except ImportError:  # pragma: no cover - supports running the file directly
    from scratch_loader import fetch_project_json


RUBRIC_FEATURES = {
    "problem_decomposition": "problem_decomposition",
    "sequencing": "sequencing",
    "loops": "loops",
    "conditionals": "conditionals",
    "variables": "variables",
    "event_handling": "event_handling",
    "debugging": "debugging",
    "procedures": "procedures",
    "coordinates": "coordinates",
    "cloning": "cloning",
    "collision": "collision",
    "animation": "animation",
    "sound": "sound",
    "ui_feedback": "ui_feedback",
    "lists": "lists",
    "math": "math",
    "messaging": "messaging",
    "algorithms": "algorithms",
    "nesting": "nesting",
    "integration": "integration",
}

RUBRIC_COLUMNS = [
    "1.Decomp",
    "2.Sequencing",
    "3.Loops",
    "4.Conditionals",
    "5.Variables",
    "6.Event Handling",
    "7.Debugging",
    "8.Procedures",
    "9.Coordinates",
    "10.Cloning",
    "11.Collision",
    "12.Animation",
    "13.Sound",
    "14.UI",
    "15.Lists",
    "16.Math",
    "17.Messaging",
    "18.Algorithms",
    "19.Nesting",
    "20.Integration",
]

RUBRIC_LABELS = {
    "1.Decomp": "problem_decomposition",
    "2.Sequencing": "sequencing",
    "3.Loops": "loops",
    "4.Conditionals": "conditionals",
    "5.Variables": "variables",
    "6.Event Handling": "event_handling",
    "7.Debugging": "debugging",
    "8.Procedures": "procedures",
    "9.Coordinates": "coordinates",
    "10.Cloning": "cloning",
    "11.Collision": "collision",
    "12.Animation": "animation",
    "13.Sound": "sound",
    "14.UI": "ui_feedback",
    "15.Lists": "lists",
    "16.Math": "math",
    "17.Messaging": "messaging",
    "18.Algorithms": "algorithms",
    "19.Nesting": "nesting",
    "20.Integration": "integration",
}


def extract_project_features(project_json):
    """Return a compact rubric-aligned feature summary for a Scratch project."""
    if not project_json:
        return {feature: False for feature in RUBRIC_FEATURES.values()}

    targets = project_json.get("targets", [])
    sprites = [target for target in targets if not target.get("isStage", False)]
    stage = next((target for target in targets if target.get("isStage", False)), None)

    block_count = 0
    opcode_counts = {}
    has_event = False
    has_loop = False
    has_conditional = False
    has_variable = False
    has_broadcast = False
    has_procedure = False
    has_coordinate = False
    has_clone = False
    has_collision = False
    has_animation = False
    has_sound = False
    has_ui_feedback = False
    has_list = False
    has_math = False
    has_debugging = False
    has_nesting = False
    has_integration = False

    for target in targets:
        blocks = target.get("blocks", {}) or {}
        for block in blocks.values():
            if not isinstance(block, dict):
                continue

            opcode = block.get("opcode", "")
            block_count += 1
            opcode_counts[opcode] = opcode_counts.get(opcode, 0) + 1

            if opcode.startswith("event_"):
                has_event = True
            if opcode.startswith("control_repeat") or opcode.startswith("control_forever"):
                has_loop = True
            if opcode.startswith("control_if") or opcode.startswith("control_if_else"):
                has_conditional = True
            if opcode.startswith("data_") or opcode.startswith("argument_"):
                has_variable = True
            if opcode.startswith("event_broadcast") or opcode.startswith("event_broadcastandwait"):
                has_broadcast = True
            if opcode.startswith("procedures_"):
                has_procedure = True
            if opcode.startswith("motion_gotoxy") or opcode.startswith("motion_glidesecstoxy"):
                has_coordinate = True
            if opcode.startswith("control_create_clone_of"):
                has_clone = True
            if opcode.startswith("sensing_touching") or opcode.startswith("sensing_touchingobject"):
                has_collision = True
            if opcode.startswith("looks_switchcostumeto") or opcode.startswith("looks_nextcostume"):
                has_animation = True
            if opcode.startswith("sound_"):
                has_sound = True
            if opcode.startswith("looks_") and any(token in opcode for token in ["say", "think", "change", "set"]):
                has_ui_feedback = True
            if opcode.startswith("data_list"):
                has_list = True
            if any(token in opcode for token in ["operator_", "math_", "data_changevariableby"]):
                has_math = True
            if opcode.startswith("procedures_call"):
                has_procedure = True

        if target.get("variables"):
            has_variable = True
        if target.get("lists"):
            has_list = True
        if target.get("sounds"):
            has_sound = True
        if target.get("costumes"):
            has_animation = True

    has_nesting = has_loop and has_conditional
    has_integration = bool(sprites and block_count >= 3 and has_event and has_variable)

    features = {
        "sprite_count": len(sprites),
        "block_count": block_count,
        "variable_count": sum(len(target.get("variables", {})) for target in targets),
        "list_count": sum(len(target.get("lists", {})) for target in targets),
        "uses_loops": has_loop,
        "uses_conditionals": has_conditional,
        "uses_variables": has_variable,
        "uses_event_handling": has_event,
        "uses_cloning": has_clone,
        "uses_collision": has_collision,
        "uses_animation": has_animation,
        "uses_sound": has_sound,
        "uses_ui_feedback": has_ui_feedback,
        "uses_lists": has_list,
        "uses_math": has_math,
        "uses_messaging": has_broadcast,
        "uses_algorithms": block_count >= 5,
        "uses_nesting": has_nesting,
        "uses_integration": has_integration,
    }
    return features


def _extract_project_id(row):
    url_value = row.get("URL") if hasattr(row, "get") else None
    if isinstance(url_value, str):
        match = re.search(r"/projects/(\d+)", url_value)
        if match:
            return int(match.group(1))

    if "ID" in row.index:
        return int(row["ID"])

    raise KeyError("No usable project ID or URL found in row")


def _extract_labels(row):
    labels = {}
    for column in RUBRIC_COLUMNS:
        value = row.get(column)
        if pd.isna(value):
            score = 0
        else:
            score = int(float(value))
        labels[RUBRIC_LABELS[column]] = score
    return labels


def enrich_dataset(df):
    enriched = []
    for _, row in df.iterrows():
        project_id = _extract_project_id(row)
        grade = row["Reviewer"]  # or whatever column holds the human grade
        project_json = fetch_project_json(project_id)
        if not project_json:
            continue

        features = extract_project_features(project_json)
        labels = _extract_labels(row)
        enriched.append({
            "id": project_id,
            "grades": labels,
            "features": features,
        })
    return enriched