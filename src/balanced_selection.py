"""Select balanced examples from dataset with least bias across grade distribution."""

from collections import defaultdict
from typing import List, Dict, Any
import random
import math


def select_balanced_examples(records: List[Dict[str, Any]], num_examples: int = 15) -> List[Dict[str, Any]]:
    """Select examples with balanced grade representation across all dimensions.
    
    Uses greedy selection to maximize coverage of grades 1-5 for each rubric dimension.
    """
    if len(records) <= num_examples:
        return records
    
    def coverage_score(selected_indices: set) -> int:
        """Score how well selected examples cover all grades 1-5 for each dimension."""
        selected_records = [records[i] for i in selected_indices]
        coverage = defaultdict(set)
        
        for record in selected_records:
            grades = record.get("grades", {})
            for dimension, grade in grades.items():
                coverage[dimension].add(grade)
        
        # Count how many dimensions have all 5 grades, weighted heavily
        perfect_dims = sum(1 for dim in coverage.values() if len(dim) == 5)
        
        # Count total unique grade-dimension combinations
        total_coverage = sum(len(v) for v in coverage.values())
        
        return perfect_dims * 100 + total_coverage
    
    # Greedy selection: pick one example at a time that maximizes coverage
    selected = set()
    for _ in range(num_examples):
        best_next = None
        best_next_score = 0
        
        for i in range(len(records)):
            if i not in selected:
                test_set = selected | {i}
                score = coverage_score(test_set)
                if score > best_next_score:
                    best_next_score = score
                    best_next = i
        
        if best_next is not None:
            selected.add(best_next)
    
    selected_indices = sorted(list(selected))
    return [records[i] for i in selected_indices]


def get_balanced_indices(records: List[Dict[str, Any]], num_examples: int = 15) -> List[int]:
    """Return indices of balanced examples."""
    if len(records) <= num_examples:
        return list(range(len(records)))
    
    def coverage_score(selected_indices: set) -> int:
        """Score how well selected examples cover all grades 1-5 for each dimension."""
        selected_records = [records[i] for i in selected_indices]
        coverage = defaultdict(set)
        
        for record in selected_records:
            grades = record.get("grades", {})
            for dimension, grade in grades.items():
                coverage[dimension].add(grade)
        
        perfect_dims = sum(1 for dim in coverage.values() if len(dim) == 5)
        total_coverage = sum(len(v) for v in coverage.values())
        return perfect_dims * 100 + total_coverage
    
    selected = set()
    for _ in range(num_examples):
        best_next = None
        best_next_score = 0
        
        for i in range(len(records)):
            if i not in selected:
                test_set = selected | {i}
                score = coverage_score(test_set)
                if score > best_next_score:
                    best_next_score = score
                    best_next = i
        
        if best_next is not None:
            selected.add(best_next)
    
    return sorted(list(selected))


def select_contrastive_examples(dataset, dimension, n_examples=6, seed=42):
	"""
	Select n_examples for a single dimension mixing low/high scores deterministically.
	- dataset: iterable of records with 'grades' dict and 'id'
	- dimension: rubric key to balance on
	- returns: list of selected records (deterministic order)
	"""
	rnd = random.Random(seed)
	buckets = defaultdict(list)
	for rec in dataset:
		score = rec.get('grades', {}).get(dimension)
		if score is None:
			continue
		buckets[score].append(rec)
	if not buckets:
		return []
	all_scores = sorted(buckets.keys())
	median_idx = len(all_scores) // 2
	low_scores = [s for s in all_scores[:median_idx+1]]
	high_scores = [s for s in all_scores[median_idx+1:]] or [all_scores[-1]]
	low_pool = [r for s in low_scores for r in buckets[s]]
	high_pool = [r for s in high_scores for r in buckets[s]]
	n_low = math.ceil(n_examples / 2)
	n_high = n_examples - n_low
	rnd.shuffle(low_pool)
	rnd.shuffle(high_pool)
	selected = (low_pool[:n_low] if low_pool else []) + (high_pool[:n_high] if high_pool else [])
	# deterministic stable ordering by id if available
	selected.sort(key=lambda r: r.get('id', 0))
	return selected


if __name__ == "__main__":
    import json
    
    with open("enriched_dataset.json") as f:
        records = json.load(f)
    
    indices = get_balanced_indices(records, num_examples=15)
    print(f"Balanced 15 example indices: {indices}")
    
    # Show coverage
    from collections import defaultdict
    coverage_by_dim = defaultdict(set)
    for idx in indices:
        record = records[idx]
        grades = record.get("grades", {})
        for dimension, grade in grades.items():
            coverage_by_dim[dimension].add(grade)
    
    print("\nGrade coverage:")
    complete = 0
    for dim in sorted(coverage_by_dim.keys()):
        grades = sorted(coverage_by_dim[dim])
        is_complete = len(grades) == 5
        if is_complete:
            complete += 1
        status = "✓" if is_complete else "✗"
        print(f"{status} {dim}: {grades}")
    
    print(f"\nDimensions with all 5 grades: {complete}/20")
    
    print(f"\nDimensions with all 5 grades: {complete}/20")
