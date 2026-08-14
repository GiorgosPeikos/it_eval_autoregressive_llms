from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IDENTITY_COLUMNS = ["component", "task_id", "fewshot", "metric"]


def compare_summaries(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in IDENTITY_COLUMNS if column in left.columns and column in right.columns]
    if "component" not in keys or "metric" not in keys:
        raise ValueError("Summaries must contain component and metric columns.")
    for name, frame in (("left", left), ("right", right)):
        if frame.duplicated(keys, keep=False).any():
            raise ValueError(f"The {name} summary has duplicate result identities for keys {keys}.")
    left = left.rename(columns={"value": "value_a", "stderr": "stderr_a"})
    right = right.rename(columns={"value": "value_b", "stderr": "stderr_b"})
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged["difference"] = merged["value_b"] - merged["value_a"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-a", required=True)
    parser.add_argument("--summary-b", required=True)
    args = parser.parse_args()
    left = pd.read_csv(args.summary_a)
    right = pd.read_csv(args.summary_b)
    merged = compare_summaries(left, right)
    output = Path(args.summary_a).resolve().parent / "comparison.csv"
    merged.to_csv(output, index=False)
    print(output)
