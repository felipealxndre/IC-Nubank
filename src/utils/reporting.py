from pathlib import Path

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def mean(xs):
    # remove None e garante float
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def save_per_query_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def aggregate_summary(per_query_rows: list[dict], k: int) -> list[dict]:
    """
    Agrupa por (agent, retriever) e calcula médias das métricas.
    Retorna lista pronta pra tabela/gráfico.
    """
    key_recall = f"recall@{k}"
    key_mrr = f"mrr@{k}"
    key_ndcg = f"ndcg@{k}"

    grouped = {}
    for r in per_query_rows:
        key = (r["agent"], r["retriever"])
        grouped.setdefault(key, []).append(r)

    summary = []
    for (agent, retriever), rs in grouped.items():
        summary.append(
            {
                "system": f"{agent}_{retriever}",
                "agent": agent,
                "retriever": retriever,
                "mean_recall": mean([x[key_recall] for x in rs]),
                "mean_mrr": mean([x[key_mrr] for x in rs]),
                "mean_ndcg": mean([x[key_ndcg] for x in rs]),
                "n_queries": len(rs),
            }
        )

    summary.sort(key=lambda x: x["mean_ndcg"], reverse=True)
    return summary


def save_summary_csv(path: Path, summary_rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)


def save_table_md(path: Path, summary_rows: list[dict], k: int):
    lines = []
    lines.append(f"| System | nDCG@{k} | MRR@{k} | Recall@{k} | #Queries |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| {r['system']} | {r['mean_ndcg']:.3f} | {r['mean_mrr']:.3f} | {r['mean_recall']:.3f} | {r['n_queries']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_barplot(path: Path, summary_rows: list[dict], metric_key: str, ylabel: str):
    labels = [r["system"] for r in summary_rows]
    values = [r[metric_key] for r in summary_rows]

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def generate_results(out_dir: Path, per_query_rows: list[dict], k: int):
    """
    Gera todos os artefatos de avaliação:
    - per_query.csv
    - summary.csv
    - table_summary.md
    - plot_ndcg.png
    - plot_mrr.png
    - plot_recall.png   
    Retorna (summary_rows, paths) para você poder logar na main.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_csv = out_dir / "per_query.csv"
    summary_csv = out_dir / "summary.csv"
    table_md = out_dir / "table_summary.md"
    plot_ndcg = out_dir / "plot_ndcg.png"
    plot_mrr = out_dir / "plot_mrr.png"

    save_per_query_csv(per_query_csv, per_query_rows)

    summary_rows = aggregate_summary(per_query_rows, k=k)
    save_summary_csv(summary_csv, summary_rows)
    save_table_md(table_md, summary_rows, k=k)

    save_barplot(plot_ndcg, summary_rows, metric_key="mean_ndcg", ylabel=f"mean nDCG@{k}")
    save_barplot(plot_mrr, summary_rows, metric_key="mean_mrr", ylabel=f"mean MRR@{k}")

    paths = {
        "per_query_csv": per_query_csv,
        "summary_csv": summary_csv,
        "table_md": table_md,
        "plot_ndcg": plot_ndcg,
        "plot_mrr": plot_mrr,
    }

    return summary_rows, paths
