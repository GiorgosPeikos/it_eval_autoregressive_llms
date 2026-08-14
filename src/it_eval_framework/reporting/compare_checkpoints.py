from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-a", required=True)
    parser.add_argument("--summary-b", required=True)
    args = parser.parse_args()
    left = pd.read_csv(args.summary_a).rename(columns={"value": "value_a"})
    right = pd.read_csv(args.summary_b).rename(columns={"value": "value_b"})
    merged = left.merge(right, on=["component", "metric"], how="outer")
    merged["difference"] = merged["value_b"] - merged["value_a"]
    output = Path(args.summary_a).resolve().parent / "comparison.csv"
    merged.to_csv(output, index=False)
    print(output)
