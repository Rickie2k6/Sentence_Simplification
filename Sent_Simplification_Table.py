import pandas as pd
import json

# Load the JSONL file uploaded by the user
file_path = "Final_Summarization.jsonl"

# Read JSONL into DataFrame
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

file_path="Final_Summarization.csv"
df.to_csv(file_path, index=False)

file_path

