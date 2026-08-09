import json
from collections import defaultdict
from pathlib import Path

def build_confusion_map(true_labels, pred_labels):
	"""
	true_labels and pred_labels are dicts keyed by sample_id -> {dim:score}
	Returns: {dim: {pred_score: mapped_score_or_adjustment}}
	"""
	# accumulate counts of (pred -> true) per-dimension
	per_dim = defaultdict(lambda: defaultdict(int))
	for sid, true_dims in (true_labels or {}).items():
		pred_dims = pred_labels.get(sid, {})
		for dim, t_val in true_dims.items():
			p_val = pred_dims.get(dim)
			per_dim[dim][(p_val, t_val)] += 1
	# convert to simple mapping: for each dim and predicted value pick most-frequent true
	mapping = {}
	for dim, pair_counts in per_dim.items():
		group = defaultdict(lambda: defaultdict(int))
		for (p_val, t_val), cnt in pair_counts.items():
			group[p_val][t_val] += cnt
		mapping[dim] = {}
		for p_val, tally in group.items():
			# choose argmax true label for this predicted value
			best_true = max(tally.items(), key=lambda x: x[1])[0]
			mapping[dim][str(p_val)] = best_true
	return mapping

def apply_confusion_remap(preds, confusion_map):
	"""
	preds: dict sample_id -> {dim:pred}
	confusion_map: output of build_confusion_map
	returns remapped preds (shallow copy)
	"""
	remapped = {}
	for sid, dims in (preds or {}).items():
		out = {}
		for dim, p in dims.items():
			m = confusion_map.get(dim, {})
			out[dim] = m.get(str(p), p)
		remapped[sid] = out
	return remapped

def save_confusion_map(confusion_map, path):
	Path(path).write_text(json.dumps(confusion_map, indent=2))

def load_confusion_map(path):
	p = Path(path)
	if not p.exists():
		return {}
	return json.loads(p.read_text())
