from dotenv import load_dotenv
import json
import os
import requests

# Load environment variables
load_dotenv()
SCRATCH_API_BASE = os.getenv("SCRATCH_API_BASE", "https://api.scratch.mit.edu/projects")
SCRATCH_PROJECTS_BASE = os.getenv("SCRATCH_PROJECTS_BASE", "https://projects.scratch.mit.edu")


def _parse_json_response(resp, project_id: int, label: str):
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type or "application/octet-stream" in content_type:
        try:
            return resp.json()
        except ValueError:
            raw_text = resp.text.strip()
            if raw_text:
                try:
                    return json.loads(raw_text)
                except ValueError as exc:
                    print(f"Warning: invalid JSON for project {project_id} ({label}): {exc}")
                    return None
    else:
        print(f"Warning: unexpected {label} response for project {project_id}: {content_type}")

    return None


def fetch_project_json(project_id: int) -> dict | None:
    """
    Fetch the full Scratch project JSON given a project ID.
    Returns a dictionary with the project data, or None when the project
    is unavailable or the request is rejected by Scratch.
    """
    metadata_url = f"{SCRATCH_API_BASE}/{project_id}"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    try:
        metadata_resp = requests.get(metadata_url, headers=headers, timeout=10)
        metadata_resp.raise_for_status()
        metadata = _parse_json_response(metadata_resp, project_id, "metadata")
        if not isinstance(metadata, dict):
            return None

        token = metadata.get("project_token")
        if token:
            project_url = f"{SCRATCH_PROJECTS_BASE}/{project_id}?token={token}"
            project_resp = requests.get(project_url, headers=headers, timeout=10)
            project_resp.raise_for_status()
            project_data = _parse_json_response(project_resp, project_id, "project")
            if isinstance(project_data, dict):
                return project_data
            return None

        return metadata
    except requests.RequestException as exc:
        print(f"Warning: could not fetch project {project_id}: {exc}")
        return None

def compute_max_nesting_depth(blocks):
	"""
	Compute max nesting depth of block dicts/lists.
	Blocks expected as list of dicts (AST-like); this is heuristic.
	"""
	max_depth = 0
	def visit(node, depth):
		nonlocal max_depth
		if depth > max_depth:
			max_depth = depth
		if isinstance(node, dict):
			for v in node.values():
				if isinstance(v, (dict, list)):
					visit(v, depth + 1)
		elif isinstance(node, list):
			for item in node:
				if isinstance(item, (dict, list)):
					visit(item, depth + 1)
	for b in (blocks or []):
		visit(b, 1)
	return max_depth

def detect_clone_events(blocks):
	"""
	Detect presence of clone create/start handlers.
	Returns dict with booleans.
	"""
	found_create = False
	found_start = False
	for b in (blocks or []):
		if isinstance(b, dict):
			op = b.get('opcode', '') or ''
			op_text = str(b.get('text','')).lower()
			if 'clone' in op_text or 'clone' in op:
				if 'create' in op_text or 'create' in op:
					found_create = True
				if 'start' in op_text or 'start_as_clone' in op:
					found_start = True
	return {"clone_create": found_create, "clone_start": found_start}

def detect_var_initializations(blocks, sample_limit=50):
	"""
	Look at early blocks for variable 'set'/'create' ops and count unique variables initialized.
	"""
	inited = set()
	count = 0
	for b in (blocks or []):
		if count >= sample_limit:
			break
		count += 1
		if isinstance(b, dict):
			op = (b.get('opcode','') or '').lower()
			if 'set' in op or 'create_variable' in op or 'init' in op:
				fields = b.get('fields', {})
				if isinstance(fields, dict):
					varname = fields.get('VAR') or fields.get('var') or None
					if varname:
						inited.add(varname)
	return {"num_var_inited": len(inited), "var_names_sample": list(inited)[:5]}

# integrate these into the atomic features pipeline where features are collected

# Optional: quick test mode
if __name__ == "__main__":
    test_id = 815  # replace with a real Scratch project ID
    data = fetch_project_json(test_id)
    print(f"Fetched project {test_id} with {len(data.keys())} top-level keys.")
