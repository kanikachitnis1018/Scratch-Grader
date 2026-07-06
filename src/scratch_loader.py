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

# Optional: quick test mode
if __name__ == "__main__":
    test_id = 815  # replace with a real Scratch project ID
    data = fetch_project_json(test_id)
    print(f"Fetched project {test_id} with {len(data.keys())} top-level keys.")
