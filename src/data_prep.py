import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ----------------------------
# TEXT CLEANING
# ----------------------------
def normalize_text(text: str) -> str:
    text = str(text).replace('\r', '\n')
    return text.strip()


# ----------------------------
# SAVE JSONL
# ----------------------------
def write_jsonl(records, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


# ----------------------------
# MAIN
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_parquet', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_parquet(args.input_parquet)

    # ----------------------------
    # CHECK COLUMNS
    # ----------------------------
    print("Available columns:", df.columns.tolist())

    if 'input' not in df.columns or 'output' not in df.columns:
        raise ValueError("Dataset must contain 'input' and 'output' columns")

    # Keep only required columns
    df = df[['input', 'output']].dropna()

    # ----------------------------
    # CLEAN TEXT
    # ----------------------------
    df['input'] = df['input'].map(normalize_text)
    df['output'] = df['output'].map(normalize_text)

    # Remove empty rows
    df = df[(df['input'] != '') & (df['output'] != '')]

    # ----------------------------
    # ANALYZE SHORT RESPONSES
    # ----------------------------
    df['output_len'] = df['output'].apply(lambda x: len(x.split()))

    short_count = (df["output_len"] < 40).sum()
    print("Number of outputs with < 40 words:", short_count)

    # ----------------------------
    # REMOVE BAD SAMPLES
    # ----------------------------
    df = df[
        (df["output_len"] >= 40) & 
        (df["output"].str.strip().str.endswith(('.', '!', '?')))
    ]

    print("After filtering:", len(df))

    # ----------------------------
    # REMOVE DUPLICATES
    # ----------------------------
    df = df.drop_duplicates(subset=['input', 'output']).reset_index(drop=True)

    # ----------------------------
    # CREATE RECORDS
    # ----------------------------
    records = [
        {
            "input": row["input"],
            "output": row["output"]
        }
        for _, row in df.iterrows()
    ]

    # ----------------------------
    # SPLIT DATA (80/10/10)
    # ----------------------------
    train_records, temp_records = train_test_split(
        records,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True
    )

    val_records, test_records = train_test_split(
        temp_records,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )

    # ----------------------------
    # SAVE FILES
    # ----------------------------
    write_jsonl(train_records, output_dir / "train.jsonl")
    write_jsonl(val_records, output_dir / "validation.jsonl")
    write_jsonl(test_records, output_dir / "test.jsonl")

    # ----------------------------
    # SUMMARY
    # ----------------------------
    summary = {
        "total_examples": len(records),
        "train": len(train_records),
        "validation": len(val_records),
        "test": len(test_records),
        "min_output_words": 40,
        "seed": args.seed
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset Summary:")
    print(json.dumps(summary, indent=2))


# ----------------------------
if __name__ == "__main__":
    main()