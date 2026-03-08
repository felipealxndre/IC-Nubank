from pathlib import Path
import re

import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def mean(xs):
    # remove None e garante float
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _format_system_label(system: str) -> str:
    """Converte 'FusionAgent_BM25Retriever' em 'Fusion Agent\n- BM25' (quebra de linha e espaços)."""
    parts = system.split("_")
    if len(parts) != 2:
        return system.replace("_", " ")
    agent, retriever = parts
    # Espaço antes de maiúsculas: FusionAgent -> Fusion Agent, StandardAgent -> Standard Agent
    agent_spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[0-9])(?=[A-Z])", " ", agent)
    # Retriever: BM25Retriever -> BM25, DenseRetriever -> Dense, Hybrid -> Hybrid
    retriever_short = retriever.replace("Retriever", "").strip() or retriever
    return f"{agent_spaced}\n- {retriever_short}"


def _apply_plot_style(ax):

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    sns.despine()
    plt.tight_layout()
    for spine in ["bottom", "left", "top", "right"]:
        ax.spines[spine].set_color("gray")
        ax.spines[spine].set_linewidth(1)


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


def save_table_as_figure(path: Path, summary_rows: list[dict], k: int):
    """Salva a tabela de resumo como imagem de um dataframe (estilo roxo)."""
    if not summary_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "System": r["system"],
                f"nDCG@{k}": f"{r['mean_ndcg']:.3f}",
                f"MRR@{k}": f"{r['mean_mrr']:.3f}",
                f"Recall@{k}": f"{r['mean_recall']:.3f}",
                "#Queries": r["n_queries"],
            }
            for r in summary_rows
        ]
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    ax.axis("off")
    n_cols = len(df.columns)
    purple_palette = sns.color_palette("Purples", n_colors=n_cols + 2)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.2)
    for j in range(n_cols):
        for (i, _) in enumerate(df.index):
            table[(i + 1, j)].set_facecolor(purple_palette[j])
        table[(0, j)].set_facecolor(purple_palette[n_cols])  # header mais escuro
        table[(0, j)].set_text_props(weight="bold", color="white")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def save_barplot(path: Path, summary_rows: list[dict], metric_key: str, ylabel: str):
    # Ordenar sempre da maior para a menor métrica para este gráfico
    sorted_rows = sorted(summary_rows, key=lambda r: r[metric_key], reverse=True)
    labels = [r["system"] for r in sorted_rows]
    labels_display = [_format_system_label(s) for s in labels]
    values = [r[metric_key] for r in sorted_rows]

    plt.rcParams["font.family"] = "Segoe UI"
    fig, ax = plt.subplots(figsize=(12, 6))
    n_bars = len(values)
    colors = sns.color_palette("Purples", n_colors=n_bars + 2)[1 : n_bars + 1][::-1]
    x_pos = range(len(values))
    ax.bar(x_pos, values, color=colors, edgecolor="gray", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_display, rotation=0, ha="center", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=20)
    plt.yticks(fontsize=14)
    _apply_plot_style(ax)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_results(out_dir: Path, per_query_rows: list[dict], k: int):
    """
    Gera todos os artefatos de avaliação:
    - per_query.csv
    - summary.csv
    - table_summary.md
    - table_summary.png (tabela como imagem)
    - plot_ndcg.png
    - plot_mrr.png
    Retorna (summary_rows, paths) para você poder logar na main.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query_csv = out_dir / "per_query.csv"
    summary_csv = out_dir / "summary.csv"
    table_md = out_dir / "table_summary.md"
    table_png = out_dir / "table_summary.png"
    plot_ndcg = out_dir / "plot_ndcg.png"
    plot_mrr = out_dir / "plot_mrr.png"

    save_per_query_csv(per_query_csv, per_query_rows)

    summary_rows = aggregate_summary(per_query_rows, k=k)
    save_summary_csv(summary_csv, summary_rows)
    save_table_md(table_md, summary_rows, k=k)
    save_table_as_figure(table_png, summary_rows, k=k)

    save_barplot(plot_ndcg, summary_rows, metric_key="mean_ndcg", ylabel=f"mean nDCG@{k}")
    save_barplot(plot_mrr, summary_rows, metric_key="mean_mrr", ylabel=f"mean MRR@{k}")

    paths = {
        "per_query_csv": per_query_csv,
        "summary_csv": summary_csv,
        "table_md": table_md,
        "table_png": table_png,
        "plot_ndcg": plot_ndcg,
        "plot_mrr": plot_mrr,
    }

    return summary_rows, paths
