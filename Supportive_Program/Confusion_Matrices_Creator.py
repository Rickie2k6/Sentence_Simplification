# Re-import necessary libraries after reset
import json
import pandas as pd
from sklearn.metrics import confusion_matrix

# Reload the data
file_path = "Final_Summarization.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Extract labels
y_true = [entry["reference_cefr"] for entry in data]     # ground truth
y_pred = [entry["predicted_cefr"] for entry in data]     # model predictions

# Build confusion matrix
labels = sorted(set(y_true) | set(y_pred))
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Convert to DataFrame
cm_df = pd.DataFrame(cm,
                     index=[f"Actual {l}" for l in labels],
                     columns=[f"Pred {l}" for l in labels])

# Save to Excel
output_path = "Confusion_Matrix.xlsx"
cm_df.to_excel(output_path, index=True)

output_path
