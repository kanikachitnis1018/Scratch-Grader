"""Select balanced examples from dataset with least bias across grade distribution."""

from collections import defaultdict
from typing import List, Dict, Any


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
