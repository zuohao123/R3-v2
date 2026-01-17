#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from glob import glob


LOG_RE = re.compile(
    r"INFO: Eval summary dataset=(?P<dataset>[a-zA-Z0-9_]+) \\| corruption=(?P<corr>[0-9.]+) "
    r"\\| EM (?P<em>[0-9.]+) F1 (?P<f1>[0-9.]+) BLEU (?P<bleu>[0-9.]+) "
    r"ROUGE-L (?P<rouge>[0-9.]+) ANLS (?P<anls>[0-9.]+) RA (?P<ra>[0-9.]+)"
)
FILE_RE = re.compile(
    r"eval_(?P<method>r3|ragonly)_(?P<dataset>[^_]+)_(?P<index>in|cross)_(?P<tag>[^.]+)\\.log$"
)


def parse_log(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "dataset": match.group("dataset"),
                    "corruption": float(match.group("corr")),
                    "EM": float(match.group("em")),
                    "F1": float(match.group("f1")),
                    "BLEU": float(match.group("bleu")),
                    "ROUGE_L": float(match.group("rouge")),
                    "ANLS": float(match.group("anls")),
                    "RA": float(match.group("ra")),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--out_csv", default="results/hard_retrieval_summary.csv")
    parser.add_argument(
        "--out_mean_csv", default="results/hard_retrieval_summary_mean.csv"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []

    for path in sorted(glob(os.path.join(args.logs_dir, "eval_*.log"))):
        name = os.path.basename(path)
        file_match = FILE_RE.match(name)
        if not file_match:
            continue
        meta = file_match.groupdict()
        parsed = parse_log(path)
        for row in parsed:
            rows.append(
                {
                    "method": meta["method"],
                    "dataset": meta["dataset"],
                    "index_mode": meta["index"],
                    "filter_tag": meta["tag"],
                    **row,
                }
            )

    if not rows:
        print("No matching logs found.")
        return

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "dataset",
            "index_mode",
            "filter_tag",
            "corruption",
            "EM",
            "F1",
            "ANLS",
            "ROUGE_L",
            "BLEU",
            "RA",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    agg = defaultdict(lambda: {"n": 0, "EM": 0.0, "F1": 0.0, "ANLS": 0.0})
    for row in rows:
        key = (row["method"], row["dataset"], row["index_mode"], row["filter_tag"])
        agg[key]["n"] += 1
        agg[key]["EM"] += row["EM"]
        agg[key]["F1"] += row["F1"]
        agg[key]["ANLS"] += row["ANLS"]

    with open(args.out_mean_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "dataset",
            "index_mode",
            "filter_tag",
            "mean_EM",
            "mean_F1",
            "mean_ANLS",
            "n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in sorted(agg.items()):
            count = value["n"]
            writer.writerow(
                {
                    "method": key[0],
                    "dataset": key[1],
                    "index_mode": key[2],
                    "filter_tag": key[3],
                    "mean_EM": round(value["EM"] / count, 4),
                    "mean_F1": round(value["F1"] / count, 4),
                    "mean_ANLS": round(value["ANLS"] / count, 4),
                    "n": count,
                }
            )

    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_mean_csv}")


if __name__ == "__main__":
    main()
