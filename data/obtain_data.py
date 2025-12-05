import csv
import json
import random
from pathlib import Path

random.seed(42)

raw_path = Path("data/raw/UKP_ASPECT.tsv")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True) 

# load tsv
rows = []
with raw_path.open("r", encoding="utf8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for r in reader:
        rows.append(r)

random.shuffle(rows)

n = len(rows) 
n_train = int(0.8 * n) # 80% for training
n_dev = int(0.1 * n) # 10% for development

splits = {
    "train": rows[:n_train],
    "dev": rows[n_train:n_train+n_dev], 
    "test": rows[n_train+n_dev:]
}

for name, items in splits.items():
    out_file = out_dir / f"{name}.jsonl"
    with out_file.open("w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
