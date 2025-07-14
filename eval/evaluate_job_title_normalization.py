import json
from datetime import datetime
from pathlib import Path
import httpx
import pandas as pd

# Load evaluation dataset from JSON
eval_dataset_path = Path("eval/eval_dataset.json")
with open(eval_dataset_path, "r") as f:
    eval_dataset = json.load(f)

# Prepare API-based predictions
api_url = "http://127.0.0.1:8000/suggestions"
threshold = 0.7
limit = 1

predictions = {}

with httpx.Client() as client:
    for row in eval_dataset:
        q = row["input_title"]
        try:
            response = client.get(
                api_url, params={"q": q, "threshold": threshold, "limit": limit}
            )
            response.raise_for_status()
            suggestions = response.json().get("suggestions", [])
            predictions[q] = suggestions[0]["name"] if suggestions else None
        except Exception as e:
            print(f"Error fetching prediction for '{q}': {e}")
            predictions[q] = None

# Build eval targets and predictions
y_true = [row["expected_title"] for row in eval_dataset]
y_pred = [predictions[row["input_title"]] for row in eval_dataset]

# Compute custom metrics directly
TP = sum((yt == yp) for yt, yp in zip(y_true, y_pred) if yp is not None)
FP = sum((yt != yp) for yt, yp in zip(y_true, y_pred) if yp is not None)
FN = sum((yp is None) for yp in y_pred)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


# Prepare report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = "all-MiniLM-L6-v2_threshold70"
report = {
    "experiment_name": experiment_name,
    "description": "Using a generic all-MiniLM-L6-v2 embedding model with a cosine similarity threshold of 0.7.",
    "timestamp": timestamp,
    "metrics": {
        "eval_samples": len(predictions.items()),
        "true_positives": TP,
        "false_positives": FP,
        "false_negatives": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    },
    "results": [
        {
            "input_title": row["input_title"],
            "expected": row["expected_title"],
            "predicted": predictions[row["input_title"]],
            "correct": predictions[row["input_title"]] == row["expected_title"],
        }
        for row in eval_dataset
    ],
}

# Save report
report_path = Path(f"eval/reports/{experiment_name}_{timestamp}.json")
report_path.parent.mkdir(parents=True, exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

# Save incorrect predictions to CSV
incorrect_results = [
    {
        "input_title": row["input_title"],
        "expected_title": row["expected_title"],
        "predicted_title": predictions[row["input_title"]],
    }
    for row in eval_dataset
    if predictions[row["input_title"]] != row["expected_title"]
]

incorrect_df = pd.DataFrame(incorrect_results)

# Construct the CSV file name from the report JSON name
incorrect_csv_path = report_path.with_name(report_path.stem + "_incorrect.csv")
incorrect_df.to_csv(incorrect_csv_path, index=False)
