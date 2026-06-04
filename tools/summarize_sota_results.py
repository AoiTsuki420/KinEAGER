import argparse
import json
from pathlib import Path

import pandas as pd


def _read_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(metrics: dict, key: str):
    v = metrics.get(key)
    if isinstance(v, (int, float)):
        return v
    return None


def _fmt(v):
    if v is None:
        return ""
    try:
        return f"{float(v):.6f}"
    except Exception:
        return ""


def build_rows(runs_dir: Path, run_specs):
    rows = []
    for model, rel_dir in run_specs:
        eval_type = "adapter" if model in {"CataPro", "CatPred", "UniKP"} else "retrained"
        metrics_path = runs_dir / rel_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append(
                {
                    "model": model,
                    "evaluation_type": eval_type,
                    "run_dir": str((runs_dir / rel_dir).as_posix()),
                    "status": "missing",
                }
            )
            continue

        m = _read_metrics(metrics_path)
        rows.append(
            {
                "model": model,
                "evaluation_type": eval_type,
                "run_dir": str((runs_dir / rel_dir).as_posix()),
                "status": "ok",
                "macro_mae": _pick(m, "macro_mae"),
                "macro_r2": _pick(m, "macro_r2"),
                "macro_pearson_r": _pick(m, "macro_pearson_r"),
                "kcat_mae": _pick(m, "kcat_mae"),
                "kcat_r2": _pick(m, "kcat_r2"),
                "kcat_pearson_r": _pick(m, "kcat_pearson_r"),
            }
        )
    return rows


def write_outputs(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    stem = "sota_core4_summary"
    csv_path = output_dir / f"{stem}.csv"
    df.to_csv(csv_path, index=False)

    md_lines = []
    md_lines.append("# SOTA Core-4 Summary")
    md_lines.append("")
    md_lines.append("| Model | Eval Type | Status | macro_mae | macro_r2 | macro_pearson_r | kcat_mae | kcat_r2 | kcat_pearson_r | Run Dir |")
    md_lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        md_lines.append(
            "| {model} | {evaluation_type} | {status} | {macro_mae} | {macro_r2} | {macro_pearson_r} | {kcat_mae} | {kcat_r2} | {kcat_pearson_r} | `{run_dir}` |".format(
                model=r.get("model", ""),
                evaluation_type=r.get("evaluation_type", ""),
                status=r.get("status", ""),
                macro_mae=_fmt(r.get("macro_mae")),
                macro_r2=_fmt(r.get("macro_r2")),
                macro_pearson_r=_fmt(r.get("macro_pearson_r")),
                kcat_mae=_fmt(r.get("kcat_mae")),
                kcat_r2=_fmt(r.get("kcat_r2")),
                kcat_pearson_r=_fmt(r.get("kcat_pearson_r")),
                run_dir=r.get("run_dir", ""),
            )
        )
    md_path = output_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize core-4 baseline runs")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    run_specs = [
        ("CataPro", "cmp_catapro_seed42/seed42"),
        ("CatPred", "cmp_catpred_seed42/seed42"),
        ("UniKP", "cmp_unikp_seed42/seed42"),
        ("DLKcat", "cmp_dlkcat_seed42/seed42"),
    ]

    rows = build_rows(runs_dir, run_specs)
    write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()
