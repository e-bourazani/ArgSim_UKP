import csv
import json
from pathlib import Path

input_file = Path("data/raw/UKP_ASPECT.tsv")
output_file = Path("data/processed/UKP_ASPECT.jsonl")
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    reader = csv.DictReader(f_in, delimiter="\t")
    for row in reader:
        f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Saved JSONL → {output_file}")
