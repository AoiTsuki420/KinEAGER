"""
tools/plot_mut_landscape.py
────────────────────────────────────────────────────────────────
突变景观可视化工具，生成以下图表：

  1. mutation_heatmap.pdf/png
     - X 轴：突变体氨基酸（20种，按物理化学分组排列）
     - Y 轴：位点（按 gate_p 降序）
     - 色彩：d_logkcat（kcat 变化量）
     - 可选：同时叠加 d_logKm 为上半图

  2. pareto_scatter.pdf/png
     - X 轴：score_kcat
     - Y 轴：score_km_beneficial（-score_km，即 Km 降低为正）
     - 点颜色：group（gate_top/control_low）
     - Pareto 前沿连线
     - 标注 top 双重有益突变

  3. convergence_curve.pdf/png
     - 迭代轮次 vs 最优累计 Δlog(kcat/Km) 分数

用法示例
────────
  python tools/plot_mut_landscape.py \
      --scan insilico_mut_scan.csv \
      --pareto pareto_front.csv \
      --out-dir figures/

  python tools/plot_mut_landscape.py \
      --scan insilico_mut_scan.csv \
      --pareto pareto_front.csv \
      --convergence results/directed_evo/convergence.csv \
      --out-dir figures/

  python tools/plot_mut_landscape.py \
      --scan insilico_mut_scan.csv \
      --out-dir figures/ \
      --format pdf \
      --only heatmap
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm



AA_ORDER = list("GAVLIMFYWSTCNQDEKRHP")  # 非极性→极性→带电→脯氨酸
AA_GROUP_COLORS = {
    "G": "#AAAAAA", "A": "#AAAAAA", "V": "#AAAAAA", "L": "#AAAAAA",
    "I": "#AAAAAA", "M": "#AAAAAA",                              # 非极性
    "F": "#5555BB", "Y": "#5555BB", "W": "#5555BB",             # 芳香族
    "S": "#55BB55", "T": "#55BB55", "C": "#55BB55",             # 极性不带电
    "N": "#88CC88", "Q": "#88CC88",
    "D": "#CC5555", "E": "#CC5555",                             # 酸性
    "K": "#5588CC", "R": "#5588CC", "H": "#5588CC",            # 碱性
    "P": "#BBAA55",                                              # 脯氨酸
}


def plot_heatmap(
    scan_df: pd.DataFrame,
    out_path: str,
    target: str = "kcat",   # "kcat" | "km"
    max_positions: int = 30,
    fmt: str = "png",
    dpi: int = 200,
):
    """
    Parameters
    ----------
    target : 'kcat' → 色彩映射 d_logkcat；'km' → d_logKm（降低为红色=好）
    """
    df = scan_df.copy()

    col     = "d_logkcat" if target == "kcat" else "d_logKm"
    label   = r"$\Delta\log k_{cat}$" if target == "kcat" else r"$\Delta\log K_m$"
    flip_km = (target == "km")

    top_pos = (
        df.groupby("pos0")["gate_p"].first()
        .sort_values(ascending=False)
        .head(max_positions)
        .index.tolist()
    )
    df = df[df["pos0"].isin(top_pos)].copy()

    pos_list = sorted(top_pos, key=lambda p: -df[df["pos0"] == p]["gate_p"].iloc[0])
    aa_list  = [a for a in AA_ORDER if a in df["mut_aa"].values]

    mat = np.full((len(pos_list), len(aa_list)), np.nan)
    for i, pos in enumerate(pos_list):
        sub = df[df["pos0"] == pos]
        for j, aa in enumerate(aa_list):
            row = sub[sub["mut_aa"] == aa]
            if len(row):
                val = float(row[col].iloc[0])
                mat[i, j] = -val if flip_km else val

    y_labels = []
    for pos in pos_list:
        sub = df[df["pos0"] == pos]
        wt  = sub["wt_aa"].iloc[0] if len(sub) else "?"
        gp  = sub["gate_p"].iloc[0] if len(sub) else 0.0
        grp = sub["group"].iloc[0] if len(sub) else ""
        marker = "★" if "gate_top" in grp else " "
        y_labels.append(f"{marker}{wt}{int(pos)+1}")

    vmax = np.nanpercentile(np.abs(mat), 95) or 0.2
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = "RdBu_r" if target == "kcat" else "RdBu"

    fig_h = max(4, len(pos_list) * 0.35 + 1.5)
    fig_w = max(8, len(aa_list) * 0.45 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(aa_list)))
    ax.set_xticklabels(aa_list, fontsize=9)
    ax.set_yticks(range(len(pos_list)))
    ax.set_yticklabels(y_labels, fontsize=8, fontfamily="monospace")

    for j, aa in enumerate(aa_list):
        ax.get_xticklabels()[j].set_color(AA_GROUP_COLORS.get(aa, "#000000"))

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=10)

    title = (f"突变效应景观 — {label}\n★=gate_top 位点 | 色彩：有益=红色")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("替换氨基酸（按物理化学性质排列）", fontsize=9)
    ax.set_ylabel("序列位点（按 gate_p 降序）", fontsize=9)

    plt.tight_layout()
    save_path = f"{out_path}.{fmt}"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] 热图 → {save_path}")



def plot_pareto_scatter(
    scan_df: pd.DataFrame,
    pareto_df: pd.DataFrame | None,
    out_path: str,
    fmt: str = "png",
    dpi: int = 200,
):
    df = scan_df.copy()

    if "score_km_beneficial" not in df.columns:
        df["score_km_beneficial"] = -df["score_km"]
    if "score_km_beneficial" not in pareto_df.columns and pareto_df is not None:
        pareto_df = pareto_df.copy()
        pareto_df["score_km_beneficial"] = -pareto_df["score_km"]

    fig, ax = plt.subplots(figsize=(7, 6))

    group_styles = {
        "gate_top":    dict(color="#E05050", alpha=0.7, s=50, marker="o", zorder=3,
                            label="gate_top"),
        "control_low": dict(color="#5080D0", alpha=0.5, s=30, marker="s", zorder=2,
                            label="control_low"),
    }
    default_style = dict(color="#888888", alpha=0.4, s=25, marker="^", zorder=1,
                         label="other")

    for grp, style in group_styles.items():
        sub = df[df["group"] == grp]
        if len(sub):
            ax.scatter(sub["score_kcat"], sub["score_km_beneficial"],
                       **{k: v for k, v in style.items() if k != "label"},
                       label=style["label"])

    other = df[~df["group"].isin(group_styles.keys())]
    if len(other):
        ax.scatter(other["score_kcat"], other["score_km_beneficial"], **default_style)

    if pareto_df is not None and len(pareto_df):
        pf = pareto_df.sort_values("score_kcat")
        ax.plot(pf["score_kcat"], pf["score_km_beneficial"],
                "k--", lw=1.2, alpha=0.6, zorder=4, label="Pareto 前沿")
        ax.scatter(pf["score_kcat"], pf["score_km_beneficial"],
                   color="gold", edgecolor="black", s=80, zorder=5, label="Pareto 候选")

        doubly = pareto_df[
            (pareto_df["score_kcat"] > 0) & (pareto_df["score_km_beneficial"] > 0)
        ].nlargest(5, "score_kcat")
        for _, r in doubly.iterrows():
            label_txt = f"{r['wt_aa']}{int(r['pos0'])+1}{r['mut_aa']}"
            ax.annotate(label_txt,
                        xy=(r["score_kcat"], r["score_km_beneficial"]),
                        xytext=(8, 4), textcoords="offset points",
                        fontsize=7.5, color="#CC2222",
                        arrowprops=dict(arrowstyle="-", color="#CC2222", lw=0.8))

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.fill_between(
        [ax.get_xlim()[0] if ax.get_xlim()[0] < 0 else -1, 10],
        0, 10,
        alpha=0.04, color="green", label="双重有益区域",
    )

    ax.set_xlabel(r"$\mathrm{score}_{kcat} = \Delta\log k_{cat} / \sigma_{kcat}$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{score}_{Km}^{benef.} = -\Delta\log K_m / \sigma_{Km}$", fontsize=11)
    ax.set_title("突变效应 Pareto 散点图\n（右上角 = kcat↑ 且 Km↓）", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = f"{out_path}.{fmt}"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] Pareto 散点图 → {save_path}")



def plot_convergence(
    conv_df: pd.DataFrame,
    out_path: str,
    fmt: str = "png",
    dpi: int = 200,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(conv_df["iter"], conv_df["best_score"],
            "o-", color="#E05050", lw=2, ms=7)
    ax.set_xlabel("迭代轮次", fontsize=11)
    ax.set_ylabel("最优综合分数", fontsize=11)
    ax.set_title("定向进化收敛曲线", fontsize=11)
    ax.grid(True, alpha=0.3)

    if "best_mut" in conv_df.columns:
        for _, row in conv_df.iterrows():
            ax.annotate(str(row["best_mut"]),
                        xy=(row["iter"], row["best_score"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7.5, color="#333333")

    ax2 = axes[1]
    ax2.bar(conv_df["iter"], conv_df["delta_score"],
            color=["#5080D0" if v >= 0 else "#CC5555" for v in conv_df["delta_score"]],
            alpha=0.75)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("迭代轮次", fontsize=11)
    ax2.set_ylabel(r"$\Delta \mathrm{score}$（本轮增量）", fontsize=11)
    ax2.set_title("边际增益（收敛判断依据）", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = f"{out_path}.{fmt}"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] 收敛曲线 → {save_path}")



def main():
    p = argparse.ArgumentParser(description="突变景观可视化")
    p.add_argument("--scan",        required=True, help="突变扫描 CSV（insilico_mut_scan.csv）")
    p.add_argument("--pareto",      default=None,  help="Pareto 前沿 CSV（可选）")
    p.add_argument("--convergence", default=None,  help="收敛曲线 CSV（convergence.csv，可选）")
    p.add_argument("--out-dir",     default="figures", help="输出目录")
    p.add_argument("--format",      default="png", choices=["png", "pdf", "svg"],
                   help="图片格式（默认 png）")
    p.add_argument("--dpi",         type=int, default=200, help="分辨率（仅 png，默认 200）")
    p.add_argument("--only",        default=None,
                   choices=["heatmap", "scatter", "convergence"],
                   help="只生成指定图表（默认全部）")
    p.add_argument("--max-pos",     type=int, default=30,
                   help="热图最多显示位点数（默认 30）")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    scan_df   = pd.read_csv(args.scan)
    pareto_df = pd.read_csv(args.pareto) if args.pareto else None

    only = args.only

    if only in (None, "heatmap"):
        plot_heatmap(
            scan_df,
            out_path=os.path.join(args.out_dir, "mutation_heatmap_kcat"),
            target="kcat",
            max_positions=args.max_pos,
            fmt=args.format, dpi=args.dpi,
        )
        plot_heatmap(
            scan_df,
            out_path=os.path.join(args.out_dir, "mutation_heatmap_km"),
            target="km",
            max_positions=args.max_pos,
            fmt=args.format, dpi=args.dpi,
        )

    if only in (None, "scatter"):
        plot_pareto_scatter(
            scan_df, pareto_df,
            out_path=os.path.join(args.out_dir, "pareto_scatter"),
            fmt=args.format, dpi=args.dpi,
        )

    if only in (None, "convergence") and args.convergence:
        conv_df = pd.read_csv(args.convergence)
        plot_convergence(
            conv_df,
            out_path=os.path.join(args.out_dir, "convergence_curve"),
            fmt=args.format, dpi=args.dpi,
        )
    elif only == "convergence" and args.convergence is None:
        print("[plot] --only convergence 需要指定 --convergence 文件")


if __name__ == "__main__":
    main()
