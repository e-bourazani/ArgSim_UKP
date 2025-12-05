import json
from pathlib import Path
import amrlib

DATA_DIR = Path("/UKP_ASPECT/data/processed")  # folder containing train/dev/test jsonl files
SPLITS = ["train", "dev", "test"]  # the jsonl files of the different splits
MODEL_DIR = "model_stog"  # full model path

# Load AMR model
stog = amrlib.load_stog_model(model_dir=MODEL_DIR)

def add_amr_to_jsonl(jsonl_file: Path):
    updated_lines = []

    # Read existing data and parse AMR
    with open(jsonl_file, "r", encoding="utf-8") as fin:
        for line in fin:
            row = json.loads(line)
            sent1 = row["sentence_1"]
            sent2 = row["sentence_2"]

            # Parse both sentences
            graphs = stog.parse_sents([sent1, sent2])
            row["amr_1"] = graphs[0] if len(graphs) > 0 else None
            row["amr_2"] = graphs[1] if len(graphs) > 1 else None

            updated_lines.append(row)

    # Overwrite the same file with added AMR entries
    with open(jsonl_file, "w", encoding="utf-8") as fout:
        for row in updated_lines:
            fout.write(json.dumps(row) + "\n")

    print(f"Updated file with AMR graphs: {jsonl_file}")

# Process all splits
for split in SPLITS:
    jsonl_file = DATA_DIR / f"{split}.jsonl"
    if jsonl_file.exists():
        add_amr_to_jsonl(jsonl_file)
    else:
        print(f"File not found: {jsonl_file}")
