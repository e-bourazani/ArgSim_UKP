import json
from pathlib import Path
import amrlib


DATA_DIR = Path("/UKP_ASPECT/data/processed")  # folder containing train/dev/test JSONL files
OUTPUT_DIR = Path("/UKP_ASPECT/data/amr_graphs")  # where to save AMR JSONL
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "dev", "test"]  # your JSONL filenames: train.jsonl, etc.
MODEL_DIR = "/UKP_ASPECT/argsim/lib/python3.10/site-packages/amrlib/data/model_stog"  # replace with your full model path

stog = amrlib.load_stog_model(model_dir=MODEL_DIR)


def parse_jsonl_pairs_to_amr(jsonl_file: Path, out_file: Path):
    with open(jsonl_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            topic = row.get("topic", "")
            sent1 = row["sentence_1"]
            sent2 = row["sentence_2"]
            label = row.get("label", None)

            # Parse both sentences
            graphs = stog.parse_sents([sent1, sent2])
            amr1 = graphs[0] if len(graphs) > 0 else None
            amr2 = graphs[1] if len(graphs) > 1 else None

            fout.write(json.dumps({
                "topic": topic,
                "sentence_1": sent1,
                "amr_1": amr1,
                "sentence_2": sent2,
                "amr_2": amr2,
                "label": label
            }) + "\n")
    print(f"Saved AMR graphs to {out_file}")


for split in SPLITS:
    jsonl_file = DATA_DIR / f"{split}.jsonl"
    out_file = OUTPUT_DIR / f"{split}_amr.jsonl"
    if jsonl_file.exists():
        parse_jsonl_pairs_to_amr(jsonl_file, out_file)
    else:
        print(f"File not found: {jsonl_file}")

