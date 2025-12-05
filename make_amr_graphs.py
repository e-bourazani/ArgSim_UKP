import json
from pathlib import Path
import amrlib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
SPLITS = ["dev"]  # for quick testing
# SPLITS = ["train", "dev", "test"]

MODEL_DIR = BASE_DIR / "models" / "model_stog" / "model_parse_t5-v0_1_0"
stog = amrlib.load_stog_model(str(MODEL_DIR))

BATCH_SIZE = 50

def process_jsonl(jsonl_file: Path):
    output_file = jsonl_file.with_suffix(".amr.jsonl")

    batch_sent1 = []
    batch_sent2 = []
    batch_rows = []

    with open(jsonl_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)
            batch_rows.append(row)
            batch_sent1.append(row["sentence_1"])
            batch_sent2.append(row["sentence_2"])

            # When reaching batch size, parse in bulk
            if len(batch_rows) >= BATCH_SIZE:
                write_batch(batch_rows, batch_sent1, batch_sent2, fout)
                print(f"Processed {len(batch_rows)} lines...")

                batch_rows = []
                batch_sent1 = []
                batch_sent2 = []

        # process remaining data
        if batch_rows:
            write_batch(batch_rows, batch_sent1, batch_sent2, fout)

    print(f"Created: {output_file}")


def write_batch(rows, sents1, sents2, fout):
    # Combine into one list of sentences
    combined = sents1 + sents2

    try:
        amrs = stog.parse_sents(combined)
    except Exception as e:
        print("AMR parsing failed for a batch:", e)
        # fallback to line-by-line parsing
        amrs = []
        for s in combined:
            try:
                g = stog.parse_sents([s])[0]
            except:
                g = None
            amrs.append(g)

    # split back into pairs
    mid = len(amrs) // 2
    amrs1 = amrs[:mid]
    amrs2 = amrs[mid:]

    # write to output safely
    for row, amr1, amr2 in zip(rows, amrs1, amrs2):
        row["amr_1"] = amr1
        row["amr_2"] = amr2
        fout.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    for split in SPLITS:
        jsonl_file = DATA_DIR / f"{split}.jsonl"
        if jsonl_file.exists():
            process_jsonl(jsonl_file)
        else:
            print(f"File not found: {jsonl_file}")