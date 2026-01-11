#!/usr/bin/env python3
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple


def _parse_list(arg: str) -> List[str]:
    if not arg:
        return []
    return [item.strip() for item in arg.split(",") if item.strip()]


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _float_key(key: str) -> float:
    try:
        return float(key)
    except ValueError:
        return float("inf")


def _format_val(val: float, scale: float, decimals: int) -> str:
    if val is None:
        return "NA"
    return f"{val * scale:.{decimals}f}"


def _collect_corruptions(dataset_blob: Dict, desired: List[str]) -> List[str]:
    keys = [k for k in dataset_blob.keys() if isinstance(dataset_blob.get(k), dict)]
    if desired:
        return [k for k in desired if k in keys]
    return sorted(keys, key=_float_key)


def _avg_metric(dataset_blob: Dict, keys: List[str], metric: str) -> float:
    vals = []
    for k in keys:
        metrics = dataset_blob.get(k, {})
        if isinstance(metrics, dict) and metric in metrics:
            vals.append(metrics[metric])
    if not vals:
        return None
    return sum(vals) / len(vals)


def _group_keys(all_keys: List[str]) -> Dict[str, List[str]]:
    # Default low/mid/high groups used in the paper.
    groups = {
        "low": ["0.0", "0.2"],
        "mid": ["0.4", "0.6"],
        "high": ["0.8", "1.0"],
        "all": all_keys,
    }
    # Keep only keys that exist for the dataset.
    return {name: [k for k in keys if k in all_keys] for name, keys in groups.items()}


def _resolve_inputs(inputs: List[str], pattern: str) -> List[str]:
    paths = []
    for item in inputs:
        paths.extend(glob.glob(item))
    if pattern:
        paths.extend(glob.glob(pattern))
    # Preserve order but de-duplicate.
    seen = set()
    dedup = []
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect eval JSONs into a text report.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="List of JSON files or glob patterns (e.g., results/*.json).",
    )
    parser.add_argument(
        "--glob",
        default="",
        help="Optional glob pattern to include (e.g., results/*corrupt*.json).",
    )
    parser.add_argument("--out", default="results/summary.txt", help="Output text file.")
    parser.add_argument(
        "--metrics",
        default="exact_match,f1,anls,rouge_l,bleu",
        help="Comma-separated metric keys to report.",
    )
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset names to include (default: all in file).",
    )
    parser.add_argument(
        "--corruptions",
        default="",
        help="Comma-separated corruption levels to include (e.g., 0.0,0.2,0.4).",
    )
    parser.add_argument("--scale", type=float, default=100.0, help="Scale factor for metrics.")
    parser.add_argument("--decimals", type=int, default=2, help="Number of decimals to print.")
    parser.add_argument(
        "--summary_metric",
        default="exact_match",
        help="Metric used for low/mid/high summary rows.",
    )
    parser.add_argument(
        "--include_meta",
        action="store_true",
        help="Include top-level meta information if present.",
    )
    args = parser.parse_args()

    metrics = _parse_list(args.metrics)
    datasets_filter = set(_parse_list(args.datasets))
    corr_filter = _parse_list(args.corruptions)

    inputs = _resolve_inputs(args.inputs, args.glob)
    if not inputs:
        raise SystemExit("No input files found. Use --inputs or --glob.")

    lines: List[str] = []
    lines.append("# Eval Summary")
    lines.append(f"Inputs: {', '.join(inputs)}")
    lines.append(f"Metrics: {', '.join(metrics)} (scale={args.scale})")
    lines.append("")

    for path in inputs:
        blob = _load_json(path)
        lines.append("=" * 80)
        lines.append(f"FILE: {path}")

        if args.include_meta and isinstance(blob.get("meta"), dict):
            lines.append("Meta:")
            for k, v in blob["meta"].items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        datasets = [k for k, v in blob.items() if isinstance(v, dict)]
        if datasets_filter:
            datasets = [d for d in datasets if d in datasets_filter]

        for dset in datasets:
            dblob = blob[dset]
            if not isinstance(dblob, dict):
                continue
            keys = _collect_corruptions(dblob, corr_filter)
            if not keys:
                continue

            lines.append(f"Dataset: {dset}")
            header = ["corruption"] + metrics
            lines.append("  " + "\t".join(header))
            for k in keys:
                row = [k]
                for m in metrics:
                    row.append(_format_val(dblob.get(k, {}).get(m), args.scale, args.decimals))
                lines.append("  " + "\t".join(row))

            # Summary with the selected metric.
            groups = _group_keys(keys)
            lines.append("  summary (mean of %s)" % args.summary_metric)
            for gname in ["low", "mid", "high", "all"]:
                gkeys = groups.get(gname, [])
                if not gkeys:
                    continue
                avg = _avg_metric(dblob, gkeys, args.summary_metric)
                lines.append(
                    "  - %s: %s"
                    % (
                        gname,
                        _format_val(avg, args.scale, args.decimals),
                    )
                )
            lines.append("")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
