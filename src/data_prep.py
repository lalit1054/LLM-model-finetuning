import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    return ' '.join(str(text).replace('\r', '\n').split())


def write_jsonl(records, path: Path) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare ChatDoctor medical QA data.')
    parser.add_argument('--input_parquet', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    df = df[['input', 'output']].dropna()
    df['input'] = df['input'].map(normalize_text)
    df['output'] = df['output'].map(normalize_text)
    df = df[(df['input'] != '') & (df['output'] != '')]
    df = df.drop_duplicates(subset=['input', 'output']).reset_index(drop=True)

    records = [
        {
            'instruction': row['input'],
            'response': row['output'],
            'source': 'chatdoctor',
        }
        for _, row in df.iterrows()
    ]

    train_records, holdout_records = train_test_split(
        records,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
    )
    validation_records, test_records = train_test_split(
        holdout_records,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True,
    )

    write_jsonl(train_records, output_dir / 'train.jsonl')
    write_jsonl(validation_records, output_dir / 'validation.jsonl')
    write_jsonl(test_records, output_dir / 'test.jsonl')

    summary = {
        'total_examples': len(records),
        'train_examples': len(train_records),
        'validation_examples': len(validation_records),
        'test_examples': len(test_records),
        'seed': args.seed,
    }
    with (output_dir / 'split_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
