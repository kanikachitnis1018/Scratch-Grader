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


RUBRIC_GUIDE = {
    "problem_decomposition": {
        "code": "TRUNK-01",
        "description": "1: No subproblems identified. 2: Incomplete/disorganized parts. 3: Clear parts described in plain language. 4: Prioritizes and maps dependencies. 5: Hierarchical, top-down planning.",
    },
    "sequencing": {
        "code": "TRUNK-02",
        "description": "1: Random/wrong order. 2: One critical step misordered. 3: Entire sequence logical and correct. 4: Efficient, avoids unnecessary steps. 5: Optimal, adaptable ordering.",
    },
    "loops": {
        "code": "TRUNK-03",
        "description": "1: Duplicates blocks instead of looping. 2: Wrong counts or infinite loops. 3: Proper repeat/repeat-until loops. 4: Chooses optimal loop types. 5: Handles edge cases; nests correctly.",
    },
    "conditionals": {
        "code": "TRUNK-04",
        "description": "1: No conditional blocks used. 2: Simple if; struggles with else/logic. 3: Correct if/else branches. 4: Uses AND/OR; handles edge cases. 5: Elegant logic, no redundancy.",
    },
    "variables": {
        "code": "TRUNK-05",
        "description": "1: All values hardcoded. 2: Uninitialized or poorly named variables. 3: Updates variables properly on events. 4: Dynamic state tracking and resets. 5: Coherent system of interdependent states.",
    },
    "event_handling": {
        "code": "TRUNK-06",
        "description": "1: Only uses green flag. 2: Misses keys or duplicates handlers. 3: Multi-event responses on correct sprites. 4: Clear, independent event architecture. 5: Coordinated flows; no timing conflicts.",
    },
    "debugging": {
        "code": "TRUNK-07",
        "description": "1: Random changes; no visible strategy. 2: Finds simple bugs; misses logic errors. 3: Displays variables, isolates code. 4: Fixes subtle logic/edge cases. 5: Methodical testing; documents via comments.",
    },
    "procedures": {
        "code": "TRUNK-08",
        "description": "1: No custom blocks ('My Blocks'). 2: Custom blocks lack parameters/consistency. 3: Parameterized, reusable custom block. 4: Library of blocks reducing duplication. 5: Elegant abstraction with default values.",
    },
    "coordinates": {
        "code": "TRUNK-09",
        "description": "1: Drag-and-drop placement only. 2: Fixed, hardcoded coordinate blocks. 3: Moves sprites absolutely and incrementally. 4: Math-based boundaries and tracking. 5: Grids or advanced movement trajectories.",
    },
    "cloning": {
        "code": "TRUNK-10",
        "description": "1: Manual sprite duplication only. 2: Clones lack independent behavior or deletion. 3: Creates and deletes clones correctly. 4: Manages lifecycles and clone limits. 5: Complex systems (waves, particles).",
    },
    "collision": {
        "code": "TRUNK-11",
        "description": "1: No interaction detection. 2: Out-of-loop checks cause missed frames. 3: Reliable sprite/edge collision events. 4: Checks sprite, edge, and color types. 5: Accounts for size, timing, and multi-outcomes.",
    },
    "animation": {
        "code": "TRUNK-12",
        "description": "1: Static sprites. 2: Poor timing or gameplay interference. 3: Smooth costume loops with wait blocks. 4: Linked to events (move, hit). 5: State-driven systems (idle, walk, jump).",
    },
    "sound": {
        "code": "TRUNK-13",
        "description": "1: No audio blocks. 2: Bad timing or overlapping tracks. 3: Event-triggered, contextual audio. 4: Manages music vs. sound effects correctly. 5: Layered, balanced audio system.",
    },
    "ui_feedback": {
        "code": "TRUNK-14",
        "description": "1: Missing score/lives/timer. 2: Cluttered, overlapping, or broken counters. 3: Real-time updates of main stats. 4: Includes start, end, and feedback screens. 5: Polished text, audio, and visual states.",
    },
    "lists": {
        "code": "TRUNK-15",
        "description": "1: Uses single variables instead of a list. 2: Hardcoded indices; no loop traversal. 3: Basic operations (add, delete, item-of). 4: Loops dynamically search/modify items. 5: Formats inventory or high-score systems.",
    },
    "math": {
        "code": "TRUNK-16",
        "description": "1: Hardcoded numbers only. 2: Simple math errors or order confusion. 3: Calculates dynamic speed, offsets, scores. 4: Fluent mod, rounding, mult, div. 5: Physics or scaling models.",
    },
    "messaging": {
        "code": "TRUNK-17",
        "description": "1: One sprite controls all; no broadcasts. 2: Single generic message for everything. 3: Named broadcasts with clear intent. 4: Descriptive, easy-to-trace architecture. 5: Protocol with no race conditions.",
    },
    "algorithms": {
        "code": "TRUNK-18",
        "description": "1: Ad-hoc logic; works by coincidence. 2: Fails on edge cases or scale. 3: Step-by-step logic works for all inputs. 4: Efficient approach chosen with reason. 5: Optimized, documented design tradeoffs.",
    },
    "nesting": {
        "code": "TRUNK-19",
        "description": "1: Flat code; no nested loops/conditionals. 2: Incorrect block scope nesting. 3: Functional nested loops or conditions. 4: Deliberate, clearly structured layers. 5: Builds state machines/complex systems.",
    },
    "integration": {
        "code": "TRUNK-20",
        "description": "1: Isolated parts; non-functional project. 2: Features present but disconnected/broken. 3: Fully playable game (systems meet baseline). 4: Polished, seamless gameplay flow. 5: Flawless mastery of all systems combined.",
    },
}

QUESTION_GUIDE = {
    "problem_decomposition": [
        "Does the project show any explicit subproblems or separated components?",
        "Are subproblems described or commented anywhere?",
        "Are boundaries between parts clear?",
        "Are dependencies between parts explicitly shown?",
        "Is there a hierarchical structure (main -> subproblems)?",
        "Are subproblems named or documented?",
    ],
    "sequencing": [
        "Are steps placed in a logical order?",
        "Does any misordered block break intended behavior?",
        "Is there a clear start-to-finish flow?",
        "Are unnecessary steps avoided?",
        "Is the ordering optimized or deliberately chosen?",
    ],
    "loops": [
        "Are loops used instead of duplicated blocks?",
        "Are repeat counts correct?",
        "Are termination conditions correct?",
        "Are loops nested?",
        "Are loop types chosen appropriately (repeat vs repeat-until)?",
        "Are edge cases handled (0 iterations, 1 iteration)?",
    ],
    "conditionals": [
        "Are if/else blocks used?",
        "Are boolean conditions correct?",
        "Are compound conditions (AND/OR) used?",
        "Are redundant checks avoided?",
        "Are multi-branch states handled?",
        "Is conditional logic clean and non-duplicated?",
    ],
    "variables": [
        "Are variables used at all?",
        "Are variables initialized correctly?",
        "Are variable names descriptive?",
        "Are multiple game states tracked?",
        "Are variables updated in response to events?",
        "Is there a coherent variable system?",
    ],
    "event_handling": [
        "Are events beyond green flag used?",
        "Are multiple event types handled (key, click, broadcast)?",
        "Are handlers attached to correct sprites?",
        "Are event names descriptive?",
        "Do sprites react independently?",
        "Are event conflicts avoided?",
    ],
    "debugging": [
        "Are debugging comments present?",
        "Are variable values displayed for debugging?",
        "Is code isolated for testing?",
        "Are edge cases tested?",
        "Are subtle logic bugs addressed?",
        "Are fixes documented?",
    ],
    "procedures": [
        "Are custom blocks used?",
        "Do custom blocks have parameters?",
        "Are custom blocks reused?",
        "Do custom blocks reduce duplication?",
        "Are block names descriptive?",
        "Are complex behaviors abstracted?",
    ],
    "coordinates": [
        "Are x/y blocks used?",
        "Are positions updated dynamically?",
        "Are incremental movements used (change x/y)?",
        "Are boundaries checked?",
        "Are positions calculated mathematically?",
        "Are coordinate systems (grids, trajectories) designed?",
    ],
    "cloning": [
        "Are clones created?",
        "Do clones behave differently from originals?",
        "Are clones deleted when no longer needed?",
        "Are clone-specific variables initialized?",
        "Is clone quantity controlled?",
        "Are complex clone systems used (waves, particles)?",
    ],
    "collision": [
        "Are touching blocks used?",
        "Is collision checked inside loops?",
        "Are multiple collision types handled?",
        "Are collisions reliable every frame?",
        "Are collision outcomes distinct?",
        "Are timing/size issues handled?",
    ],
    "animation": [
        "Are costume changes used?",
        "Is animation timing correct?",
        "Do animations loop smoothly?",
        "Are animations tied to game events?",
        "Are multiple animation states used?",
        "Are transitions between states smooth?",
    ],
    "sound": [
        "Are sounds used?",
        "Are sounds triggered by events?",
        "Is timing correct (no overlap)?",
        "Is background music separated from SFX?",
        "Are start-sound vs play-until-done used correctly?",
        "Is there a layered sound system?",
    ],
    "ui_feedback": [
        "Is score/lives/timer shown?",
        "Does UI update in real time?",
        "Is UI clear and readable?",
        "Are start/end screens present?",
        "Are feedback messages used?",
        "Is UI polished and consistent?",
    ],
    "lists": [
        "Are lists used?",
        "Are list items accessed dynamically?",
        "Are list operations used (add/delete/item-of)?",
        "Are lists processed with loops?",
        "Are lists updated during gameplay?",
        "Are list-based systems designed (inventory, high scores)?",
    ],
    "math": [
        "Are arithmetic operators used?",
        "Are dynamic values computed?",
        "Are advanced operators used (multiply/divide/modulo)?",
        "Are operators chosen correctly?",
        "Are mathematical models used (scaling, physics)?",
        "Are random distributions or formulas used?",
    ],
    "messaging": [
        "Are broadcasts used?",
        "Are messages distinctly named?",
        "Do sprites respond correctly?",
        "Are multi-sprite interactions coordinated?",
        "Is messaging architecture clear?",
        "Are circular dependencies avoided?",
    ],
    "algorithms": [
        "Is there a step-by-step strategy?",
        "Does the algorithm work for all expected inputs?",
        "Are edge cases handled?",
        "Is efficiency considered?",
        "Are alternative approaches evaluated?",
        "Are algorithms documented?",
    ],
    "nesting": [
        "Are loops nested?",
        "Are conditionals nested?",
        "Is nesting correct (no scope errors)?",
        "Does nesting serve a clear purpose?",
        "Are nested structures documented?",
        "Are nested systems used (state machines, patterns)?",
    ],
    "integration": [
        "Do major systems (movement, collision, scoring, UI) work together?",
        "Is the project fully playable?",
        "Are features consistent across sprites?",
        "Are skills integrated smoothly?",
        "Is the project polished?",
        "Does the whole system demonstrate mastery?",
    ],
}


SELECTIVE_COT_DIMENSIONS = {
    "variables",
    "math",
    "lists",
    "sound",
    "collision",
    "event_handling",
    "messaging",
    "procedures",
    "nesting",
    "ui_feedback",
}

HYBRID_EVIDENCE_RULES = """
EVIDENCE-FIRST GRADING PROCESS (must follow):
1) Infer evidence from features first.
2) Then assign scores 0-5 from evidence.
3) Be conservative when evidence is missing.

Weak dimensions to evaluate carefully:
- problem_decomposition
- sequencing
- animation
- algorithms
- nesting

High-score gates:
- algorithms=5 only if ALL are evidenced: step-by-step strategy, works across inputs, edge-case handling, efficiency awareness.
- animation=5 only if ALL are evidenced: costume/sprite changes, smooth looping, event-linked transitions.
- sequencing>=4 only if logical order and clear start-to-finish flow are evidenced.
- problem_decomposition>=4 only if clear subproblems plus hierarchy/dependencies are evidenced.
- nesting>=4 only if nesting is purposeful and not incidental.
"""

FINAL_ONLY_INSTRUCTION = """
Return ONLY valid JSON.
No markdown. No prose. No explanation.
Include all 20 rubric keys with integer values 0-5 only.
"""


def _format_rubric_guide() -> str:
    """Format the complete rubric guide for inclusion in the prompt."""
    lines = ["SCORING GUIDE - Rubric Dimensions (1=Novice to 5=Expert):\n"]
    for dimension, info in RUBRIC_GUIDE.items():
        lines.append(f"{info['code']}: {dimension.upper()}")
        lines.append(f"{info['description']}\n")
    return "\n".join(lines)


def _format_question_guide() -> str:
    """Format the question-based rubric guide for inclusion in the prompt."""
    lines = ["QUESTION GUIDE (Levels 1-5):\n"]
    for index, (dimension, questions) in enumerate(QUESTION_GUIDE.items(), start=1):
        lines.append(f"{index}. {dimension.replace('_', ' ').title()}")
        for question in questions:
            lines.append(f"- {question}")
        lines.append("")
    lines.append("Score mapping: 1 = no skill, 2 = partial skill, 3 = correct skill, 4 = efficient skill, 5 = expert skill.")
    return "\n".join(lines)

RUBRIC_CATEGORIES = {
    "core_logic": ["problem_decomposition", "sequencing", "loops", "conditionals", "algorithms", "nesting"],
    "state_data": ["variables", "lists", "math"],
    "interactivity_events": ["event_handling", "messaging", "ui_feedback"],
    "media_physics": ["coordinates", "cloning", "collision", "animation", "sound"],
    "engineering_quality": ["debugging", "procedures", "integration"],
}

RUBRIC_SUMMARY = {
    "problem_decomposition": "1=no decomposition | 2=partial/incomplete | 3=clear manageable parts | 4=systematic with dependencies | 5=hierarchical top-down",
    "sequencing": "1=random/incorrect | 2=mostly correct with one misorder | 3=correct logical flow | 4=efficient deliberate | 5=optimal adaptable",
    "loops": "1=duplicates code | 2=simple repeat misused | 3=correct repeat-until | 4=appropriate type combined | 5=optimized nested with edge cases",
    "conditionals": "1=no conditionals | 2=basic if only | 3=correct if/else | 4=multiple AND/OR | 5=elegant multi-branch",
    "variables": "1=none | 2=declared misused | 3=correct storage updates | 4=multiple descriptive with reset | 5=coherent multi-state",
    "event_handling": "1=green flag only | 2=one or two events missed | 3=correct multi-event | 4=independent event architecture | 5=coordinated no conflicts",
    "debugging": "1=random changes | 2=finds obvious bugs | 3=systematic isolation | 4=subtle logic errors | 5=methodical testing documented",
    "procedures": "1=no custom blocks | 2=blocks lack parameters | 3=parameterized reusable | 4=library reducing duplication | 5=elegant abstraction",
    "coordinates": "1=drag-drop only | 2=fixed hardcoded | 3=absolute incremental | 4=math-based boundaries | 5=grids advanced trajectories",
    "cloning": "1=manual duplication | 2=clones lack behavior | 3=creates deletes correctly | 4=manages lifecycles | 5=complex systems waves",
    "collision": "1=no detection | 2=out-of-loop checks | 3=reliable sprite/edge | 4=sprite edge color | 5=size timing outcomes",
    "animation": "1=static sprites | 2=poor timing | 3=smooth costume loops | 4=linked to events | 5=state-driven idle/walk/jump",
    "sound": "1=no audio | 2=bad timing overlapping | 3=event-triggered contextual | 4=manages music vs effects | 5=layered balanced",
    "ui_feedback": "1=missing score/lives | 2=cluttered overlapping | 3=real-time main stats | 4=start end screens | 5=polished combined states",
    "lists": "1=uses single variables | 2=hardcoded indices | 3=add delete item-of | 4=dynamic search/modify loops | 5=inventory high-score formats",
    "math": "1=hardcoded only | 2=simple errors | 3=dynamic speed offsets scores | 4=fluent mod rounding mult div | 5=physics scaling models",
    "messaging": "1=one sprite all control | 2=single generic message | 3=named broadcasts | 4=descriptive architecture | 5=protocol no race",
    "algorithms": "1=ad-hoc coincidence | 2=fails edge cases | 3=step-by-step all inputs | 4=efficient reasoned | 5=optimized documented",
    "nesting": "1=flat no nesting | 2=incorrect scope | 3=functional nested | 4=deliberate structured | 5=state machines complex",
    "integration": "1=isolated non-functional | 2=features broken disconnected | 3=fully playable baseline | 4=polished seamless | 5=flawless mastery",
}

SELECTIVE_COT_DIMENSIONS = {
    "variables",
    "math",
    "lists",
    "sound",
    "collision",
    "event_handling",
    "messaging",
    "procedures",
    "nesting",
    "ui_feedback",
}

HYBRID_EVIDENCE_RULES = """
EVIDENCE-FIRST GRADING PROCESS (must follow):
1) Infer evidence from features first.
2) Then assign scores 0-5 from evidence.
3) Be conservative when evidence is missing.

Weak dimensions to evaluate carefully:
- problem_decomposition
- sequencing
- animation
- algorithms
- nesting

High-score gates:
- algorithms=5 only if ALL are evidenced: step-by-step strategy, works across inputs, edge-case handling, efficiency awareness.
- animation=5 only if ALL are evidenced: costume/sprite changes, smooth looping, event-linked transitions.
- sequencing>=4 only if logical order and clear start-to-finish flow are evidenced.
- problem_decomposition>=4 only if clear subproblems plus hierarchy/dependencies are evidenced.
- nesting>=4 only if nesting is purposeful and not incidental.
"""

FINAL_ONLY_INSTRUCTION = """
Return ONLY valid JSON.
No markdown. No prose. No explanation.
Include all 20 rubric keys with integer values 0-5 only.
"""


def _format_rubric_summary() -> str:
    """Format compact rubric summaries for all 20 dimensions."""
    lines = ["RUBRIC SUMMARY (Levels 1–5):\n"]
    for dimension, summary in RUBRIC_SUMMARY.items():
        lines.append(f"• {dimension}: {summary}")
    return "\n".join(lines)


def _generate_selective_reasoning(features: Dict[str, Any]) -> str:
    """Generate reasoning hints for feature-driven dimensions without score leakage."""
    steps = []
    
    sprite_count = features.get("sprite_count", 0)
    block_count = features.get("block_count", 0)
    variable_count = features.get("variable_count", 0)
    list_count = features.get("list_count", 0)

    if "variables" in SELECTIVE_COT_DIMENSIONS:
        var_reasoning = (
            "variables: "
            f"{variable_count} variable(s) defined. "
        )
        if variable_count > 0:
            var_reasoning += "Variables are present and likely used for state tracking. " if features.get("uses_variables") else "Variables declared but usage unclear. "
            var_reasoning += "Score reflects proper initialization, updates on events, and reset logic."
        else:
            var_reasoning += "No variables → level 1."
        steps.append(f"  - {var_reasoning}")

    if "math" in SELECTIVE_COT_DIMENSIONS:
        math_reasoning = "math: "
        if features.get("uses_math"):
            math_reasoning += "Math operators detected. Score based on complexity (simple arithmetic→3, division/modulo→4, physics models→5)."
        else:
            math_reasoning += "No math operators → level 1 (hardcoded only)."
        steps.append(f"  - {math_reasoning}")

    if "lists" in SELECTIVE_COT_DIMENSIONS:
        lists_reasoning = "lists: "
        if list_count > 0:
            lists_reasoning += f"{list_count} list(s) present. Score based on: hardcoded indices→2, add/delete operations→3, dynamic loops→4, inventory systems→5."
        else:
            lists_reasoning += "No lists defined → level 1 (single variables)."
        steps.append(f"  - {lists_reasoning}")

    if "sound" in SELECTIVE_COT_DIMENSIONS:
        sound_reasoning = "sound: "
        if features.get("uses_sound"):
            sound_reasoning += "Sound blocks detected. Score based on timing, context (event-triggered→3), music vs effects separation→4, layering→5."
        else:
            sound_reasoning += "No audio blocks → level 1."
        steps.append(f"  - {sound_reasoning}")

    if "collision" in SELECTIVE_COT_DIMENSIONS:
        collision_reasoning = "collision: "
        if features.get("uses_collision"):
            collision_reasoning += "Collision blocks detected. Score based on: out-of-loop→2, reliable in loops→3, sprite/edge/color checks→4, size/timing accounting→5."
        else:
            collision_reasoning += "No collision detection → level 1."
        steps.append(f"  - {collision_reasoning}")

    if "event_handling" in SELECTIVE_COT_DIMENSIONS:
        event_reasoning = "event_handling: "
        if features.get("uses_event_handling"):
            event_reasoning += "Event blocks detected. Score should reflect whether the project goes beyond green flag, coordinates multiple triggers, and assigns events across sprites rather than centralising all control."
        else:
            event_reasoning += "No event blocks beyond basic startup evidence → low score unless other trigger evidence exists."
        steps.append(f"  - {event_reasoning}")

    if "messaging" in SELECTIVE_COT_DIMENSIONS:
        messaging_reasoning = "messaging: "
        if features.get("uses_messaging"):
            messaging_reasoning += "Broadcast blocks detected. Score based on whether messages seem distinct, support sprite coordination, and avoid a single generic broadcast for all state changes."
        else:
            messaging_reasoning += "No broadcast blocks → level 1."
        steps.append(f"  - {messaging_reasoning}")

    if "procedures" in SELECTIVE_COT_DIMENSIONS:
        procedures_reasoning = "procedures: "
        if features.get("uses_algorithms") and features.get("block_count", 0) >= 30:
            procedures_reasoning += "Larger projects should only score highly here if custom reusable blocks are truly present and reduce duplication; project size alone should not inflate the score."
        else:
            procedures_reasoning += "Only score above the baseline when there is clear evidence of custom reusable abstractions."
        steps.append(f"  - {procedures_reasoning}")

    if "nesting" in SELECTIVE_COT_DIMENSIONS:
        nesting_reasoning = "nesting: "
        if features.get("uses_nesting"):
            nesting_reasoning += "Loops and conditionals both appear. Only assign high scores if control structures are meaningfully nested rather than merely coexisting in separate scripts."
        else:
            nesting_reasoning += "No evidence of nested control structures → low score."
        steps.append(f"  - {nesting_reasoning}")

    if "ui_feedback" in SELECTIVE_COT_DIMENSIONS:
        ui_reasoning = "ui_feedback: "
        if features.get("uses_ui_feedback"):
            ui_reasoning += "Say/think/visual feedback exists. Score should distinguish simple feedback from a maintained score/lives/timer UI and proper start/end states."
        else:
            ui_reasoning += "No visible UI feedback blocks → low score unless other explicit UI evidence is present."
        steps.append(f"  - {ui_reasoning}")

    return "\n".join(steps)


def build_prompt_hybrid(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """Build hybrid prompt: evidence-first + rubric summaries + examples + selective COT."""
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    required_keys = list(RUBRIC_SUMMARY.keys())
    blocks = [
        _format_rubric_summary(),
        HYBRID_EVIDENCE_RULES.strip(),
        f"Required JSON keys (exact, all must be present): {json.dumps(required_keys)}",
    ]

    for index, record in enumerate(examples, start=1):
        block = "\n".join([
            f"Example {index}:",
            "Features:",
            _format_features(record["features"]),
            "Grades:",
            _format_grades(record["grades"]),
        ])
        blocks.append(block)

    reasoning = _generate_selective_reasoning(target["features"])
    target_block = "\n".join([
        "Now grade this new project:",
        "Features:",
        _format_features(target["features"]),
        "Reasoning (for feature-driven dimensions):",
        reasoning,
        "Use balanced scoring: low only when evidence is absent, mid when partial, high when multiple signals agree.",
    ])
    blocks.append(target_block)
    blocks.append(FINAL_ONLY_INSTRUCTION.strip())
    return "\n\n".join(blocks)


def build_prompt_rubric_reference(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """Build a few-shot prompt with full rubric reference guide.

    This variant includes the complete scoring guide at the beginning, allowing
    the model to reference exact rubric definitions and level descriptions while
    grading the target project.
    """
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    blocks = [_format_rubric_guide()]

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
        "Now grade this new project using the rubric guide above:",
        "Features:",
        _format_features(target["features"]),
    ])
    blocks.append(target_block)
    blocks.append("Return only the grades as a JSON object with the same rubric keys, referring to the scoring guide for each dimension.")
    return "\n\n".join(blocks)


def build_prompt_question_based(records: List[Dict[str, Any]], num_examples: int = 3) -> str:
    """Build a few-shot prompt using the question set instead of the rubric guide."""
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    blocks = [_format_question_guide()]

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
        "Now grade this new project using the question guide above:",
        "Features:",
        _format_features(target["features"]),
    ])
    blocks.append(target_block)
    blocks.append("Return only the grades as a JSON object with the same rubric keys, using the question guide and the score mapping above.")
    return "\n\n".join(blocks)

def build_prompt_category_batch(
    records: List[Dict[str, Any]], 
    category_name: str, 
    num_examples: int = 3,
    prompt_style: str = "hybrid"  # Default or set to hybrid
) -> str:
    """Build a prompt evaluating ONLY dimensions within a specific category batch."""
    if category_name not in RUBRIC_CATEGORIES:
        raise ValueError(f"Unknown category: {category_name}")

    target_dims = RUBRIC_CATEGORIES[category_name]
    if not records:
        raise ValueError("records must not be empty")

    examples = records[:max(1, min(num_examples, len(records) - 1))]
    target = records[min(len(records) - 1, num_examples)] if len(records) > num_examples else records[-1]

    blocks = [f"CATEGORY: {category_name.upper()}\nFocus ONLY on these dimensions: {', '.join(target_dims)}\n"]

    # Filter compact rubric summaries for current batch dimensions
    category_summaries = [f"• {dim}: {RUBRIC_SUMMARY[dim]}" for dim in target_dims if dim in RUBRIC_SUMMARY]
    blocks.append("RUBRIC SUMMARY:\n" + "\n".join(category_summaries))

    # Add few-shot examples
    for index, record in enumerate(examples, start=1):
        filtered_grades = {k: v for k, v in record["grades"].items() if k in target_dims}
        block = "\n".join([
            f"Example {index}:",
            "Features:",
            _format_features(record["features"]),
            "Grades:",
            _format_grades(filtered_grades),
        ])
        blocks.append(block)

    # Add selective CoT reasoning if prompt_style is hybrid
    target_block_lines = [
        "Now grade this new project:",
        "Features:",
        _format_features(target["features"]),
    ]
    
    if prompt_style == "hybrid":
        reasoning = _generate_selective_reasoning(target["features"])
        # Keep reasoning lines relevant to target_dims
        filtered_reasoning = "\n".join([
            line for line in reasoning.split("\n") 
            if any(dim in line for dim in target_dims)
        ])
        if filtered_reasoning.strip():
            target_block_lines.extend([
                "Reasoning (for feature-driven dimensions in this category):",
                filtered_reasoning
            ])

    blocks.append("\n".join(target_block_lines))
    blocks.append(
        f"Return ONLY a JSON object with integer scores (0–5) for these exact keys: {json.dumps(target_dims)}. Do NOT include reasoning in the JSON."
    )
    return "\n\n".join(blocks)


def save_prompt_to_file(records: List[Dict[str, Any]], output_path: str, num_examples: int = 3) -> str:
    prompt = build_few_shot_prompt(records, num_examples=num_examples)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(prompt)
    return prompt

def build_prompt(example_list, rubric_schema, selective_reasoning=None, enforce_json=True):
	"""
	- example_list: few-shot examples (contrastive preferred)
	- rubric_schema: dict describing keys and allowed values
	- selective_reasoning: list e.g. ['algorithms','sequencing','integration']
	- enforce_json: append strict schema + 'RETURN ONLY JSON' suffix
	"""
	prompt_sections = []
	# header
	prompt_sections.append("You are an expert in grading student projects. Carefully evaluate the features and provide grades.")

	# few-shot examples
	for index, example in enumerate(example_list, start=1):
		example_features = example["features"]
		example_grades = example["grades"]

		# format features and grades
		formatted_features = _format_features(example_features)
		formatted_grades = _format_grades(example_grades)

		# example block
		example_block = f"Example {index}:\nFeatures:\n{formatted_features}\nGrades:\n{formatted_grades}"
		prompt_sections.append(example_block)

	# selective reasoning snippets
	if selective_reasoning:
		for block in selective_reasoning:
			if block == "algorithms":
				prompt_sections.append("Selective reasoning (algorithms): If relevant, briefly identify core algorithmic patterns (e.g., loops, conditionals, event flow) in one line.")
			elif block == "sequencing":
				prompt_sections.append("Selective reasoning (sequencing): If relevant, list key ordered steps the project executes (1-3 bullets).")
			elif block == "integration":
				prompt_sections.append("Selective reasoning (integration): If relevant, note how sprites/variables communicate or reuse code.")
	# strict JSON enforcement
	if enforce_json:
		# concise schema reminder
		prompt_sections.append("RETURN ONLY JSON matching this schema: " + str(rubric_schema))
		prompt_sections.append("If you cannot produce valid JSON, return {\"error\":\"parse_failed\",\"raw\":<full_response>}")

	return "\n\n".join(prompt_sections)

def _selective_reasoning_snippets(selective_reasoning):
	parts = []
	if not selective_reasoning:
		return parts
	for block in selective_reasoning:
		if block == "algorithms":
			parts.append("Selective reasoning (algorithms): briefly identify core algorithmic patterns (loops, conditionals, event flow).")
		elif block == "sequencing":
			parts.append("Selective reasoning (sequencing): list key ordered steps the project executes (1-3 bullets).")
		elif block == "integration":
			parts.append("Selective reasoning (integration): note sprite/variable communication or code reuse.")
	return parts

def build_prompt_variant(examples, rubric_schema, facts=None, selective_reasoning=None, enforce_json=True, variant_id=0):
	"""
	Build one prompt variant. Use variant_id to slightly change wording for ensembling.
	- examples: list of few-shot examples (strings)
	- rubric_schema: dict schema for JSON output
	- facts: optional list of short "Fact: ..." strings to anchor model
	- selective_reasoning: list of reasoning blocks to include
	"""
	parts = []
	if facts:
		parts.append("Facts: " + " | ".join(facts))
	# small wording variants encourage diverse but focused responses
	if variant_id == 0:
		parts.append("You are a careful rubric grader. Use the examples to infer scores.")
	elif variant_id == 1:
		parts.append("Act as an objective grader. Rely on the examples and the facts above.")
	else:
		parts.append("Follow the rubric; prefer conservative judgments when uncertain.")
	# examples
	parts.append("Examples:\n" + "\n\n".join(examples))
	# selective reasoning
	parts.extend(_selective_reasoning_snippets(selective_reasoning))
	# strict JSON enforcement
	if enforce_json:
		parts.append("RETURN ONLY JSON matching this schema:\n" + str(rubric_schema))
		parts.append("If you cannot produce valid JSON, return {\"error\":\"parse_failed\",\"raw\":<full_response>}")
	return "\n\n".join(parts)

def build_prompt_ensemble(examples, rubric_schema, facts=None, selective_reasoning=None, n_variants=3):
	"""
	Return list of prompt strings (n_variants) for ensembling/majority voting.
	"""
	return [build_prompt_variant(examples, rubric_schema, facts=facts, selective_reasoning=selective_reasoning, variant_id=i) for i in range(n_variants)]
