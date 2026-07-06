from dotenv import load_dotenv
import os
import pandas as pd
import json
from grader import enrich_dataset   # make sure this import is here

def load_dataset():
    load_dotenv()
    dataset_path = os.getenv("DATASET_PATH")
    df = pd.read_excel(dataset_path)
    print(f"Loaded {len(df)} graded projects")
    return df

if __name__ == "__main__":
    df = load_dataset()
    enriched = enrich_dataset(df)

    valid_projects = [entry for entry in enriched if entry.get("features")]
    print(f"Processed {len(valid_projects)} usable projects out of {len(df)} rows")

    # Save only the filtered set of usable projects
    with open("enriched_dataset.json", "w") as f:
        json.dump(valid_projects, f, indent=2)

    print("Saved enriched dataset to enriched_dataset.json")
    if valid_projects:
        print("First enriched project:", valid_projects[0])
    else:
        print("No usable projects were found.")
