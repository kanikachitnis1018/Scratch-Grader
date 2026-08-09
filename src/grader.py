import json
import re
from pathlib import Path

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

ATOMIC_QUESTION_KEYS = {
    "problem_decomposition": [
        "has_subproblems",
        "has_hierarchy",
        "comments_describe_subproblems",
        "clear_boundaries",
        "dependencies_explicit",
        "subproblems_named_or_documented",
    ],
    "sequencing": [
        "logical_order",
        "misordered_block_breaks_behavior",
        "clear_start_to_finish_flow",
        "unnecessary_steps_avoided",
        "optimized_order",
    ],
    "loops": [
        "loops_used_instead_of_duplication",
        "repeat_counts_correct",
        "termination_conditions_correct",
        "loops_nested",
        "loop_types_appropriate",
        "edge_cases_handled",
    ],
    "conditionals": [
        "if_else_used",
        "boolean_conditions_correct",
        "compound_conditions_used",
        "redundant_checks_avoided",
        "multi_branch_states_handled",
        "conditional_logic_clean",
    ],
    "variables": [
        "variables_used",
        "variables_initialized",
        "variable_names_descriptive",
        "multiple_states_tracked",
        "variables_updated_on_events",
        "coherent_variable_system",
    ],
    "event_handling": [
        "events_beyond_green_flag",
        "multiple_event_types_handled",
        "handlers_attached_to_correct_sprites",
        "event_names_descriptive",
        "sprites_react_independently",
        "event_conflicts_avoided",
    ],
    "debugging": [
        "debugging_comments_present",
        "variable_values_displayed_for_debugging",
        "code_isolated_for_testing",
        "edge_cases_tested",
        "subtle_logic_bugs_addressed",
        "fixes_documented",
    ],
    "procedures": [
        "custom_blocks_used",
        "custom_blocks_have_parameters",
        "custom_blocks_reused",
        "custom_blocks_reduce_duplication",
        "block_names_descriptive",
        "complex_behaviors_abstracted",
    ],
    "coordinates": [
        "x_y_blocks_used",
        "positions_updated_dynamically",
        "incremental_movements_used",
        "boundaries_checked",
        "positions_calculated_mathematically",
        "coordinate_systems_designed",
    ],
    "cloning": [
        "clones_created",
        "clones_behave_differently_from_originals",
        "clones_deleted_when_no_longer_needed",
        "clone_specific_variables_initialized",
        "clone_quantity_controlled",
        "complex_clone_systems_used",
    ],
    "collision": [
        "touching_blocks_used",
        "collision_checked_inside_loops",
        "multiple_collision_types_handled",
        "collisions_reliable_every_frame",
        "collision_outcomes_distinct",
        "timing_size_issues_handled",
    ],
    "animation": [
        "costume_changes_used",
        "animation_timing_correct",
        "animations_loop_smoothly",
        "animations_tied_to_game_events",
        "multiple_animation_states_used",
        "transitions_smooth",
    ],
    "sound": [
        "sounds_used",
        "sounds_triggered_by_events",
        "timing_correct_no_overlap",
        "background_music_separated_from_sfx",
        "start_sound_vs_play_until_done_correct",
        "layered_sound_system",
    ],
    "ui_feedback": [
        "score_lives_timer_shown",
        "ui_updates_in_real_time",
        "ui_clear_and_readable",
        "start_end_screens_present",
        "feedback_messages_used",
        "ui_polished_and_consistent",
    ],
    "lists": [
        "lists_used",
        "list_items_accessed_dynamically",
        "list_operations_used",
        "lists_processed_with_loops",
        "lists_updated_during_gameplay",
        "list_based_systems_designed",
    ],
    "math": [
        "arithmetic_operators_used",
        "dynamic_values_computed",
        "advanced_operators_used",
        "operators_chosen_correctly",
        "mathematical_models_used",
        "random_or_formula_used",
    ],
    "messaging": [
        "broadcasts_used",
        "messages_differently_named",
        "sprites_respond_correctly",
        "multi_sprite_interactions_coordinated",
        "messaging_architecture_clear",
        "circular_dependencies_avoided",
    ],
    "algorithms": [
        "step_by_step_strategy",
        "algorithm_works_for_inputs",
        "edge_cases_handled",
        "efficiency_considered",
        "alternative_approaches_evaluated",
        "algorithms_documented",
    ],
    "nesting": [
        "loops_nested",
        "conditionals_nested",
        "nesting_correct",
        "nesting_serves_clear_purpose",
        "nested_structures_documented",
        "nested_systems_used",
    ],
    "integration": [
        "major_systems_work_together",
        "project_fully_playable",
        "features_consistent_across_sprites",
        "skills_integrated_smoothly",
        "project_polished",
        "whole_system_demonstrates_mastery",
    ],
}

DEFAULT_DUMPS_DIR = Path(__file__).resolve().parent.parent / "dumps"
RAW_PROJECTS_FILE = "raw_projects.json"
ATOMIC_FEATURES_FILE = "atomic_features.json"
MODEL_SCORES_FILE = "model_scores.json"


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


def _collect_project_context(project_json):
    targets = project_json.get("targets", []) if project_json else []
    sprites = [target for target in targets if not target.get("isStage", False)]

    opcodes = []
    target_blocks = []
    event_types = set()
    broadcast_names = set()
    handler_sprite_names = set()
    conditional_count = 0
    if_else_count = 0
    nested_control_count = 0
    procedure_definition_count = 0
    procedure_call_count = 0
    costume_switch_count = 0
    wait_block_count = 0
    loop_count = 0
    operator_count = 0
    broadcast_send_count = 0
    broadcast_receive_count = 0
    max_control_nesting_depth = 0

    def _control_ancestor_depth(blocks_by_id, block):
        depth = 0
        parent_id = block.get("parent")
        while parent_id and parent_id in blocks_by_id:
            parent_block = blocks_by_id.get(parent_id) or {}
            parent_opcode = parent_block.get("opcode", "")
            if parent_opcode.startswith(("control_repeat", "control_forever", "control_repeat_until", "control_if", "control_if_else")):
                depth += 1
            parent_id = parent_block.get("parent")
        return depth

    for target in targets:
        sprite_name = target.get("name", "")
        blocks = target.get("blocks", {}) or {}
        for _, block in blocks.items():
            if isinstance(block, dict):
                opcode = block.get("opcode", "")
                opcodes.append(opcode)
                target_blocks.append(block)
                if opcode.startswith("event_when"):
                    event_types.add(opcode)
                    if sprite_name:
                        handler_sprite_names.add(sprite_name)
                if opcode.startswith("event_broadcast") or opcode.startswith("event_whenbroadcastreceived"):
                    fields = block.get("fields", {}) or {}
                    option = fields.get("BROADCAST_OPTION")
                    if isinstance(option, list) and option:
                        broadcast_names.add(str(option[0]))
                if opcode.startswith("event_broadcast"):
                    broadcast_send_count += 1
                if opcode.startswith("event_whenbroadcastreceived"):
                    broadcast_receive_count += 1
                if opcode.startswith("control_if"):
                    conditional_count += 1
                if opcode.startswith("control_if_else"):
                    if_else_count += 1
                if opcode.startswith(("control_repeat", "control_forever", "control_repeat_until")):
                    loop_count += 1
                if opcode.startswith("procedures_definition"):
                    procedure_definition_count += 1
                if opcode.startswith("procedures_call"):
                    procedure_call_count += 1
                if opcode.startswith(("looks_switchcostumeto", "looks_nextcostume")):
                    costume_switch_count += 1
                if opcode.startswith("control_wait"):
                    wait_block_count += 1
                if opcode.startswith("operator_"):
                    operator_count += 1

                max_control_nesting_depth = max(
                    max_control_nesting_depth,
                    _control_ancestor_depth(blocks, block),
                )

                parent_id = block.get("parent")
                if parent_id and parent_id in blocks:
                    parent_block = blocks.get(parent_id) or {}
                    parent_opcode = parent_block.get("opcode", "")
                    if parent_opcode.startswith(("control_repeat", "control_forever", "control_if", "control_if_else")) and opcode.startswith(("control_repeat", "control_forever", "control_if", "control_if_else")):
                        nested_control_count += 1

    variable_names = []
    list_names = []
    sprite_names = []
    for target in targets:
        name = target.get("name")
        if isinstance(name, str) and name:
            sprite_names.append(name)

        for value in (target.get("variables") or {}).values():
            if isinstance(value, list) and value:
                variable_names.append(str(value[0]))
        for value in (target.get("lists") or {}).values():
            if isinstance(value, list) and value:
                list_names.append(str(value[0]))

    comments = []
    for target in targets:
        for comment in (target.get("comments") or {}).values():
            if isinstance(comment, dict):
                text = comment.get("text", "")
                if text:
                    comments.append(text)

    return {
        "targets": targets,
        "sprites": sprites,
        "blocks": target_blocks,
        "opcodes": opcodes,
        "variable_names": variable_names,
        "list_names": list_names,
        "sprite_names": sprite_names,
        "comments": comments,
        "event_types": event_types,
        "broadcast_names": broadcast_names,
        "handler_sprite_names": handler_sprite_names,
        "conditional_count": conditional_count,
        "if_else_count": if_else_count,
        "nested_control_count": nested_control_count,
        "procedure_definition_count": procedure_definition_count,
        "procedure_call_count": procedure_call_count,
        "costume_switch_count": costume_switch_count,
        "wait_block_count": wait_block_count,
        "loop_count": loop_count,
        "operator_count": operator_count,
        "broadcast_send_count": broadcast_send_count,
        "broadcast_receive_count": broadcast_receive_count,
        "max_control_nesting_depth": max_control_nesting_depth,
    }


def extract_atomic_features(project_json):
    """Return nested boolean answers for the 20 rubric dimensions."""
    context = _collect_project_context(project_json)
    targets = context["targets"]
    sprites = context["sprites"]
    opcodes = context["opcodes"]
    variable_names = context["variable_names"]
    list_names = context["list_names"]
    sprite_names = context["sprite_names"]
    comments = context["comments"]
    event_types = context["event_types"]
    broadcast_names = context["broadcast_names"]
    handler_sprite_names = context["handler_sprite_names"]
    conditional_count = context["conditional_count"]
    if_else_count = context["if_else_count"]
    nested_control_count = context["nested_control_count"]
    procedure_definition_count = context["procedure_definition_count"]
    procedure_call_count = context["procedure_call_count"]
    costume_switch_count = context["costume_switch_count"]
    wait_block_count = context["wait_block_count"]
    loop_count = context["loop_count"]
    operator_count = context["operator_count"]
    broadcast_send_count = context["broadcast_send_count"]
    broadcast_receive_count = context["broadcast_receive_count"]
    max_control_nesting_depth = context["max_control_nesting_depth"]

    block_count = len(opcodes)
    has_event = any(opcode.startswith("event_") for opcode in opcodes)
    has_loop = any(opcode.startswith("control_repeat") or opcode.startswith("control_forever") for opcode in opcodes)
    has_conditional = any(opcode.startswith("control_if") or opcode.startswith("control_if_else") for opcode in opcodes)
    has_variable = any(opcode.startswith("data_") or opcode.startswith("argument_") for opcode in opcodes) or bool(variable_names)
    has_broadcast = any(opcode.startswith("event_broadcast") or opcode.startswith("event_broadcastandwait") for opcode in opcodes)
    has_procedure = any(opcode.startswith("procedures_") or opcode.startswith("procedures_call") for opcode in opcodes)
    has_coordinate = any(opcode.startswith("motion_gotoxy") or opcode.startswith("motion_glidesecstoxy") for opcode in opcodes)
    has_clone = any(opcode.startswith("control_create_clone_of") for opcode in opcodes)
    has_collision = any(opcode.startswith("sensing_touching") or opcode.startswith("sensing_touchingobject") for opcode in opcodes)
    has_animation = any(opcode.startswith("looks_switchcostumeto") or opcode.startswith("looks_nextcostume") for opcode in opcodes)
    has_sound = any(opcode.startswith("sound_") for opcode in opcodes)
    has_ui_feedback = any(opcode.startswith("looks_") and any(token in opcode for token in ["say", "think", "change", "set"]) for opcode in opcodes)
    has_list = any(opcode.startswith("data_list") for opcode in opcodes) or bool(list_names)
    has_math = any(token in opcode for opcode in opcodes for token in ["operator_", "math_"])
    has_nesting = nested_control_count > 0
    coordinate_ops = {opcode for opcode in opcodes if opcode.startswith(("motion_gotoxy", "motion_glidesecstoxy", "motion_changexby", "motion_changeyby", "motion_setx", "motion_sety"))}
    has_integration = False

    variable_names_descriptive = any(name and not name.lower().startswith(("var", "item", "list")) for name in variable_names)
    list_names_descriptive = any(name and not name.lower().startswith(("list", "item")) for name in list_names)
    sprite_names_descriptive = any(name and name.lower() not in {"sprite1", "sprite2", "sprite3", "stage"} for name in sprite_names)
    compound_conditionals = any(opcode in {"operator_and", "operator_or", "operator_not"} for opcode in opcodes)
    ui_variable_names = {"score", "scores", "lives", "life", "timer", "time", "health", "hp"}
    has_ui_state_variable = any(name.lower() in ui_variable_names for name in variable_names)
    has_start_end_screen_name = any(token in name.lower() for token in sprite_names for token in ["start", "end", "game over", "menu"]) if sprite_names else False
    sends_broadcast = any(opcode.startswith("event_broadcast") for opcode in opcodes)
    receives_broadcast = any(opcode.startswith("event_whenbroadcastreceived") for opcode in opcodes)
    has_edge_bounce = any(opcode == "motion_ifonedgebounce" for opcode in opcodes)
    list_update_opcodes = {
        "data_addtolist",
        "data_deleteoflist",
        "data_deletealloflist",
        "data_insertatlist",
        "data_replaceitemoflist",
    }
    list_access_opcodes = {
        "data_itemoflist",
        "data_itemnumoflist",
        "data_lengthoflist",
        "data_listcontainsitem",
    }
    has_list_updates = any(opcode in list_update_opcodes for opcode in opcodes)
    has_dynamic_list_access = any(opcode in list_access_opcodes for opcode in opcodes)

    def _is_descriptive_broadcast_name(name: str) -> bool:
        lowered = name.strip().lower()
        if not lowered:
            return False
        generic_names = {
            "message1",
            "message2",
            "message3",
            "message4",
            "message5",
            "broadcast",
            "event",
        }
        return lowered not in generic_names

    descriptive_broadcast_names = {name for name in broadcast_names if _is_descriptive_broadcast_name(name)}

    atomic = {
        "problem_decomposition": {
            "has_subproblems": len(sprites) > 1,
            "has_hierarchy": has_procedure or has_broadcast,
            "comments_describe_subproblems": any("subproblem" in comment.lower() or "component" in comment.lower() for comment in comments),
            "clear_boundaries": len(sprites) > 1 or has_procedure,
            "dependencies_explicit": has_broadcast or has_procedure,
            "subproblems_named_or_documented": sprite_names_descriptive,
        },
        "sequencing": {
            "logical_order": block_count > 0,
            "misordered_block_breaks_behavior": False,
            "clear_start_to_finish_flow": has_event,
            "unnecessary_steps_avoided": has_loop or has_conditional or has_broadcast,
            "optimized_order": False,
        },
        "loops": {
            "loops_used_instead_of_duplication": has_loop,
            "repeat_counts_correct": False,
            "termination_conditions_correct": False,
            "loops_nested": has_nesting,
            "loop_types_appropriate": False,
            "edge_cases_handled": False,
        },
        "conditionals": {
            "if_else_used": if_else_count > 0,
            "boolean_conditions_correct": conditional_count > 0,
            "compound_conditions_used": compound_conditionals and conditional_count >= 2,
            "redundant_checks_avoided": False,
            "multi_branch_states_handled": if_else_count > 0,
            "conditional_logic_clean": conditional_count >= 2,
        },
        "variables": {
            "variables_used": has_variable,
            "variables_initialized": bool(variable_names),
            "variable_names_descriptive": variable_names_descriptive,
            "multiple_states_tracked": len(variable_names) > 1,
            "variables_updated_on_events": has_event and has_variable,
            "coherent_variable_system": len(variable_names) > 0 and variable_names_descriptive,
        },
        "event_handling": {
            "events_beyond_green_flag": any(event_type != "event_whenflagclicked" for event_type in event_types),
            "multiple_event_types_handled": len(event_types) >= 2,
            "handlers_attached_to_correct_sprites": len(handler_sprite_names) >= 2,
            "event_names_descriptive": len(descriptive_broadcast_names) >= 1,
            "sprites_react_independently": len(handler_sprite_names) >= 2,
            "event_conflicts_avoided": (
                broadcast_send_count >= 1
                and broadcast_receive_count >= 1
                and broadcast_receive_count >= broadcast_send_count
                and broadcast_receive_count <= (broadcast_send_count * 2)
                and 2 <= len(event_types) <= 6
            ),
        },
        "debugging": {
            "debugging_comments_present": any("debug" in comment.lower() for comment in comments),
            "variable_values_displayed_for_debugging": False,
            "code_isolated_for_testing": False,
            "edge_cases_tested": False,
            "subtle_logic_bugs_addressed": False,
            "fixes_documented": any("fix" in comment.lower() for comment in comments),
        },
        "procedures": {
            "custom_blocks_used": has_procedure,
            "custom_blocks_have_parameters": procedure_definition_count > 0,
            "custom_blocks_reused": procedure_call_count >= 2,
            "custom_blocks_reduce_duplication": procedure_call_count >= 2,
            "block_names_descriptive": procedure_definition_count > 0,
            "complex_behaviors_abstracted": procedure_definition_count > 0 and procedure_call_count > 0,
        },
        "coordinates": {
            "x_y_blocks_used": bool(coordinate_ops & {"motion_gotoxy", "motion_glidesecstoxy", "motion_setx", "motion_sety"}),
            "positions_updated_dynamically": bool(coordinate_ops),
            "incremental_movements_used": bool(coordinate_ops & {"motion_changexby", "motion_changeyby"}),
            "boundaries_checked": has_edge_bounce,
            "positions_calculated_mathematically": has_math and bool(coordinate_ops),
            "coordinate_systems_designed": False,
        },
        "cloning": {
            "clones_created": has_clone,
            "clones_behave_differently_from_originals": False,
            "clones_deleted_when_no_longer_needed": False,
            "clone_specific_variables_initialized": False,
            "clone_quantity_controlled": False,
            "complex_clone_systems_used": False,
        },
        "collision": {
            "touching_blocks_used": has_collision,
            "collision_checked_inside_loops": has_collision and has_loop,
            "multiple_collision_types_handled": False,
            "collisions_reliable_every_frame": False,
            "collision_outcomes_distinct": False,
            "timing_size_issues_handled": False,
        },
        "animation": {
            "costume_changes_used": has_animation,
            "animation_timing_correct": costume_switch_count > 0 and wait_block_count > 0,
            "animations_loop_smoothly": costume_switch_count >= 2 and loop_count > 0,
            "animations_tied_to_game_events": has_animation and has_event,
            "multiple_animation_states_used": costume_switch_count >= 3,
            "transitions_smooth": costume_switch_count >= 3 and wait_block_count > 0,
        },
        "sound": {
            "sounds_used": has_sound,
            "sounds_triggered_by_events": has_sound and has_event,
            "timing_correct_no_overlap": False,
            "background_music_separated_from_sfx": False,
            "start_sound_vs_play_until_done_correct": False,
            "layered_sound_system": False,
        },
        "ui_feedback": {
            "score_lives_timer_shown": has_ui_state_variable,
            "ui_updates_in_real_time": has_ui_state_variable and has_event,
            "ui_clear_and_readable": has_ui_feedback or has_ui_state_variable,
            "start_end_screens_present": has_start_end_screen_name,
            "feedback_messages_used": any(opcode.startswith(("looks_say", "looks_think")) for opcode in opcodes),
            "ui_polished_and_consistent": False,
        },
        "lists": {
            "lists_used": has_list,
            "list_items_accessed_dynamically": has_dynamic_list_access,
            "list_operations_used": has_list,
            "lists_processed_with_loops": has_list and has_loop,
            "lists_updated_during_gameplay": has_list_updates,
            "list_based_systems_designed": list_names_descriptive,
        },
        "math": {
            "arithmetic_operators_used": has_math,
            "dynamic_values_computed": has_math,
            "advanced_operators_used": False,
            "operators_chosen_correctly": False,
            "mathematical_models_used": False,
            "random_or_formula_used": False,
        },
        "messaging": {
            "broadcasts_used": has_broadcast,
            "messages_differently_named": len(descriptive_broadcast_names) >= 2,
            "sprites_respond_correctly": (
                sends_broadcast and receives_broadcast and broadcast_receive_count >= broadcast_send_count
            ),
            "multi_sprite_interactions_coordinated": (
                len(sprites) > 1
                and len(handler_sprite_names) >= 2
                and broadcast_send_count >= 2
                and broadcast_receive_count >= 2
            ),
            "messaging_architecture_clear": (
                len(descriptive_broadcast_names) >= 2
                and broadcast_send_count >= 2
                and broadcast_receive_count >= 2
            ),
            "circular_dependencies_avoided": (
                sends_broadcast
                and receives_broadcast
                and broadcast_send_count >= 2
                and broadcast_receive_count >= 2
                and broadcast_receive_count <= (broadcast_send_count * 2)
            ),
        },
        "algorithms": {
            "step_by_step_strategy": block_count >= 10,
            "algorithm_works_for_inputs": has_loop and has_conditional,
            "edge_cases_handled": if_else_count > 0,
            "efficiency_considered": loop_count > 0 and operator_count > 0,
            "alternative_approaches_evaluated": False,
            "algorithms_documented": False,
        },
        "nesting": {
            "loops_nested": max_control_nesting_depth >= 2 and has_loop,
            "conditionals_nested": max_control_nesting_depth >= 2 and has_conditional,
            "nesting_correct": max_control_nesting_depth >= 2,
            "nesting_serves_clear_purpose": (
                max_control_nesting_depth >= 3
                and nested_control_count >= 5
                and has_loop
                and has_conditional
            ),
            "nested_structures_documented": False,
            "nested_systems_used": max_control_nesting_depth >= 4 and nested_control_count >= 10,
        },
        "integration": {
            "major_systems_work_together": False,
            "project_fully_playable": False,
            "features_consistent_across_sprites": False,
            "skills_integrated_smoothly": False,
            "project_polished": False,
            "whole_system_demonstrates_mastery": False,
        },
    }

    return atomic


def map_scores(atomic_dict):
    """Map atomic yes/no answers to final 1-5 scores for each dimension."""
    scores = {}
    for dimension, answers in atomic_dict.items():
        if not isinstance(answers, dict) or not answers:
            scores[dimension] = 1
            continue

        true_count = sum(1 for value in answers.values() if bool(value))
        if dimension in {"debugging", "integration"}:
            if true_count == 0:
                score = 0
            else:
                score = min(2, true_count)
        elif dimension == "conditionals":
            if not answers.get("boolean_conditions_correct"):
                score = 1
            elif not answers.get("if_else_used"):
                score = 2
            elif answers.get("compound_conditions_used") and answers.get("conditional_logic_clean"):
                score = 4
            elif answers.get("multi_branch_states_handled"):
                score = 3
            else:
                score = 3
        elif dimension == "variables":
            if not answers.get("variables_used"):
                score = 1
            elif not answers.get("variables_initialized"):
                score = 2
            elif answers.get("multiple_states_tracked") and answers.get("variables_updated_on_events"):
                if answers.get("coherent_variable_system"):
                    score = 4
                else:
                    score = 3
            elif answers.get("variable_names_descriptive"):
                score = 3
            else:
                score = 2
        elif dimension == "event_handling":
            if not answers.get("events_beyond_green_flag") or not answers.get("handlers_attached_to_correct_sprites"):
                score = 1
            elif not answers.get("multiple_event_types_handled") or not answers.get("sprites_react_independently"):
                score = 2
            elif not answers.get("event_names_descriptive") or not answers.get("event_conflicts_avoided"):
                score = 3
            elif true_count == 6:
                score = 5
            elif true_count >= 5:
                score = 4
            else:
                score = 3
        elif dimension == "messaging":
            if not answers.get("broadcasts_used"):
                score = 1
            elif not answers.get("sprites_respond_correctly"):
                score = 2
            elif answers.get("messaging_architecture_clear") and answers.get("multi_sprite_interactions_coordinated"):
                if answers.get("circular_dependencies_avoided") and answers.get("messages_differently_named") and true_count == 6:
                    score = 5
                else:
                    score = 4
            elif answers.get("messages_differently_named"):
                score = 3
            else:
                score = 2
        elif dimension == "ui_feedback":
            if true_count == 0:
                score = 1
            elif answers.get("start_end_screens_present") and answers.get("score_lives_timer_shown"):
                score = 4
            elif true_count >= 3:
                score = 3
            else:
                score = 2
        elif dimension == "coordinates":
            if not answers.get("positions_updated_dynamically"):
                score = 1
            elif answers.get("positions_calculated_mathematically") or answers.get("boundaries_checked"):
                score = 4
            else:
                score = 3
        elif dimension == "problem_decomposition":
            if not answers.get("has_subproblems"):
                score = 1
            elif answers.get("has_hierarchy") and answers.get("dependencies_explicit"):
                score = 4
            elif answers.get("clear_boundaries"):
                score = 3
            else:
                score = 2
        elif dimension == "nesting":
            if not answers.get("loops_nested") and not answers.get("conditionals_nested"):
                score = 1
            elif not answers.get("nesting_correct") or not answers.get("nesting_serves_clear_purpose"):
                score = 2
            elif answers.get("nested_systems_used"):
                score = 4
            elif answers.get("loops_nested") and answers.get("conditionals_nested"):
                score = 3
            else:
                score = 2
        elif dimension == "procedures":
            if not answers.get("custom_blocks_used"):
                score = 1
            elif answers.get("custom_blocks_reused") and answers.get("complex_behaviors_abstracted"):
                score = 4
            elif answers.get("custom_blocks_have_parameters"):
                score = 3
            else:
                score = 2
        elif dimension == "animation":
            if not answers.get("costume_changes_used"):
                score = 1
            elif answers.get("multiple_animation_states_used") and answers.get("transitions_smooth"):
                score = 4
            elif answers.get("animation_timing_correct") or answers.get("animations_loop_smoothly"):
                score = 3
            else:
                score = 2
        elif dimension == "algorithms":
            if not answers.get("step_by_step_strategy"):
                score = 1
            elif not answers.get("algorithm_works_for_inputs"):
                score = 2
            elif answers.get("algorithm_works_for_inputs") and answers.get("efficiency_considered") and answers.get("edge_cases_handled"):
                score = 4
            elif answers.get("algorithm_works_for_inputs"):
                score = 3
            else:
                score = 2
        elif dimension == "lists":
            if not answers.get("lists_used"):
                score = 1
            elif not answers.get("list_operations_used"):
                score = 2
            elif answers.get("lists_processed_with_loops") and answers.get("lists_updated_during_gameplay"):
                if answers.get("list_items_accessed_dynamically") and answers.get("list_based_systems_designed"):
                    score = 5
                else:
                    score = 4
            elif answers.get("list_items_accessed_dynamically") or answers.get("lists_updated_during_gameplay"):
                score = 3
            else:
                score = 2
        else:
            total = len(answers)
            if true_count == 0:
                score = 1
            else:
                ratio = true_count / total
                if ratio <= 0.20:
                    score = 2
                elif ratio <= 0.40:
                    score = 3
                elif ratio <= 0.60:
                    score = 4
                else:
                    score = 5
        scores[dimension] = score

    return scores


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _save_stage_outputs(raw_projects, atomic_features_by_project, model_scores_by_project, output_dir=None):
    base_dir = Path(output_dir) if output_dir else DEFAULT_DUMPS_DIR
    raw_path = base_dir / RAW_PROJECTS_FILE
    atomic_path = base_dir / ATOMIC_FEATURES_FILE
    score_path = base_dir / MODEL_SCORES_FILE

    _write_json(raw_path, raw_projects)
    _write_json(atomic_path, atomic_features_by_project)
    _write_json(score_path, model_scores_by_project)

    return {
        "raw_projects_path": str(raw_path),
        "atomic_features_path": str(atomic_path),
        "model_scores_path": str(score_path),
    }


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


def enrich_dataset(df, output_dir=None):
    enriched = []
    raw_projects = {}
    atomic_features_by_project = {}
    model_scores_by_project = {}

    for _, row in df.iterrows():
        try:
            project_id = _extract_project_id(row)
        except KeyError as exc:
            print(f"Warning: skipping row without usable project id/url: {exc}")
            continue

        project_json = fetch_project_json(project_id)
        if not project_json:
            continue

        try:
            atomic_features = extract_atomic_features(project_json)
            model_scores = map_scores(atomic_features)
        except OSError as exc:
            print(f"Warning: could not prepare stage artifacts for project {project_id}: {exc}")
            continue

        project_key = str(project_id)
        raw_projects[project_key] = project_json
        atomic_features_by_project[project_key] = atomic_features
        model_scores_by_project[project_key] = model_scores

        features = extract_project_features(project_json)
        labels = _extract_labels(row)
        enriched.append({
            "id": project_id,
            "grades": labels,
            "features": features,
            "atomic_features": atomic_features,
            "model_scores": model_scores,
        })

    if enriched:
        try:
            stage_paths = _save_stage_outputs(
                raw_projects,
                atomic_features_by_project,
                model_scores_by_project,
                output_dir=output_dir,
            )
        except OSError as exc:
            print(f"Warning: could not save debug artifacts: {exc}")
        else:
            for record in enriched:
                record.update(stage_paths)
    return enriched


def apply_hard_constraints(features, preds):
    """
    Deterministic post-processing clamps for model predictions.

    - features: compact feature dict returned by extract_project_features OR
      a dict that also contains an "atomic_features" key (the atomic map).
    - preds: dict mapping rubric-dimension -> numeric score (ints or floats).
    Returns a new dict (does not mutate input preds).
    """
    out = dict(preds or {})

    # Accept either the compact features dict or a record that embeds atomic_features
    atomic = None
    if isinstance(features, dict):
        atomic = features.get("atomic_features") if isinstance(features.get("atomic_features"), dict) else None

    # convenience getters
    def feat(key, default=None):
        try:
            return features.get(key, default) if isinstance(features, dict) else default
        except Exception:
            return default

    def atomic_bool(dim, key):
        try:
            return bool(atomic.get(dim, {}).get(key))
        except Exception:
            return False

    # basic signals
    block_count = int(feat("block_count") or 0)
    uses_animation = bool(feat("uses_animation")) or atomic_bool("animation", "costume_changes_used")
    uses_cloning = bool(feat("uses_cloning")) or atomic_bool("cloning", "clones_created")
    uses_event = bool(feat("uses_event_handling")) or atomic_bool("sequencing", "clear_start_to_finish_flow") or atomic_bool("event_handling", "events_beyond_green_flag")
    uses_variables = bool(feat("uses_variables")) or atomic_bool("variables", "variables_used")
    uses_nesting = bool(feat("uses_nesting")) or atomic_bool("nesting", "nesting_correct")
    max_nesting = feat("max_control_nesting_depth") if feat("max_control_nesting_depth") is not None else None
    if max_nesting is None and atomic:
        # try to infer from nesting atomic flags
        max_nesting = 4 if atomic_bool("nesting", "nested_systems_used") else (2 if (atomic_bool("nesting", "loops_nested") or atomic_bool("nesting", "conditionals_nested")) else 0)

    # 1) force obvious zeros / lows
    if not uses_cloning and "cloning" in out:
        out["cloning"] = 0

    if not uses_animation and "animation" in out:
        out["animation"] = 0

    if not uses_event and "event_handling" in out:
        out["event_handling"] = 0

    if not uses_variables and "variables" in out:
        out["variables"] = 1

    if not feat("uses_integration", False) and atomic:
        integration_atomic = atomic.get("integration", {})
        if integration_atomic and not any(bool(v) for v in integration_atomic.values()) and "integration" in out:
            out["integration"] = 0

    # 2) small-project downgrades for algorithms/sequencing
    if block_count < 3:
        if "algorithms" in out:
            out["algorithms"] = min(int(out.get("algorithms", 1)), 1)
        if "sequencing" in out:
            out["sequencing"] = min(int(out.get("sequencing", 1)), 1)
    elif block_count < 7:
        # unlikely to be full 5-level algorithm project
        if "algorithms" in out and int(round(float(out.get("algorithms", 1)))) == 5:
            out["algorithms"] = 4
        if "sequencing" in out and int(round(float(out.get("sequencing", 1)))) == 5:
            out["sequencing"] = 4

    # 3) sequencing: require logical_order or event-based flow to be high
    if "sequencing" in out:
        seq_score = int(round(float(out.get("sequencing", 1))))
        if seq_score >= 4:
            has_logical = atomic_bool("sequencing", "logical_order")
            has_flow = atomic_bool("sequencing", "clear_start_to_finish_flow")
            if not (has_logical or has_flow):
                out["sequencing"] = 3

        if seq_score == 1 and (has_flow := (uses_event or atomic_bool("sequencing", "clear_start_to_finish_flow"))):
            out["sequencing"] = 2

    # 4) problem_decomposition: demote if no subproblems or hierarchy evidence
    if "problem_decomposition" in out:
        pd_score = int(round(float(out.get("problem_decomposition", 1))))
        has_sub = atomic_bool("problem_decomposition", "has_subproblems")
        has_hier = atomic_bool("problem_decomposition", "has_hierarchy")
        if not has_sub:
            out["problem_decomposition"] = min(pd_score, 2)
        elif pd_score >= 4 and not (has_hier and atomic_bool("problem_decomposition", "dependencies_explicit")):
            out["problem_decomposition"] = max(3, min(pd_score, 4))

    # 5) nesting: use explicit max_nesting or atomic nesting booleans
    if "nesting" in out:
        nest_score = int(round(float(out.get("nesting", 1))))
        depth = int(max_nesting or 0)
        if depth <= 1:
            out["nesting"] = min(nest_score, 1)
        elif depth == 2:
            out["nesting"] = min(nest_score, 3)
        # if nesting atomic indicates purpose absent, lower high scores
        if nest_score >= 4 and not atomic_bool("nesting", "nesting_serves_clear_purpose"):
            out["nesting"] = min(out["nesting"], 3)

    # 6) animation: require multiple animation signals for top scores
    if "animation" in out:
        anim_score = int(round(float(out.get("animation", 1))))
        has_costume = atomic_bool("animation", "costume_changes_used")
        smooth = atomic_bool("animation", "animations_loop_smoothly")
        tied = atomic_bool("animation", "animations_tied_to_game_events")
        if not has_costume:
            out["animation"] = 0
        elif anim_score == 5 and not (smooth and tied):
            out["animation"] = 4
        elif anim_score == 4 and not (smooth or tied):
            out["animation"] = 3

    # 7) algorithms: require multiple atomic signals for very high scores
    if "algorithms" in out:
        alg_score = int(round(float(out.get("algorithms", 1))))
        step = atomic_bool("algorithms", "step_by_step_strategy")
        works = atomic_bool("algorithms", "algorithm_works_for_inputs")
        eff = atomic_bool("algorithms", "efficiency_considered")
        edge = atomic_bool("algorithms", "edge_cases_handled")
        true_support = sum([step, works, eff, edge])
        if block_count < 10 and alg_score >= 5:
            out["algorithms"] = 4
        if alg_score >= 4 and true_support < 3:
            out["algorithms"] = max(3, min(out["algorithms"], 4))
        if not step and alg_score >= 3:
            out["algorithms"] = min(out["algorithms"], 2)

    # 8) guard against implausible 5s across several weak dims
    for dim in ("algorithms", "animation", "sequencing", "problem_decomposition", "nesting"):
        if dim in out and int(round(float(out.get(dim, 0)))) == 5:
            # count supporting atomic truths (if available)
            support = 0
            if atomic and isinstance(atomic.get(dim), dict):
                support = sum(1 for v in atomic.get(dim).values() if bool(v))
            # if few supports, demote to 4 (or 3 if very few)
            if support <= 2:
                out[dim] = 4 if support == 2 else 3

    # Normalize and ensure integer 0-5
    for k, v in list(out.items()):
        try:
            iv = int(round(float(v)))
        except Exception:
            iv = 0
        out[k] = max(0, min(5, iv))

    return out

# Integration hint:
# Call apply_hard_constraints(project_features, raw_preds) after parsing model output and before final scoring.