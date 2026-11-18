"""FactLens: Publication-Grade Report Figures Generator.

Reads benchmark results from `results/benchmark_metrics.json` and produces 3
high-resolution (300 DPI) publication-quality figures for the IEEE technical report:
  1. report/figures/fig1_pipeline_architecture.png
  2. report/figures/fig2_ablation_comparison.png
  3. report/figures/fig3_category_breakdown.png
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


# Set publication style defaults
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def generate_fig1_pipeline_architecture(output_path: Path) -> None:
    """Generate high-DPI academic block diagram for the 4-stage FactLens pipeline."""
    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Color Palette (Pastel Academic / Clean IEEE standard)
    c_input = "#F1F5F9"
    c_stage1 = "#E0F2FE"  # Sky Blue
    c_stage2 = "#EDE9FE"  # Purple
    c_stage3 = "#FEF3C7"  # Amber
    c_stage4 = "#DCFCE7"  # Emerald
    c_output = "#F8FAFC"
    border_c = "#475569"

    # Input Block: Left column
    # 1. Source Document
    box_src = patches.FancyBboxPatch(
        (0.3, 2.9), 1.9, 1.7, boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=c_input, edgecolor=border_c, linewidth=1.2
    )
    ax.add_patch(box_src)
    ax.text(1.25, 4.25, "Source Document", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
    ax.text(1.25, 3.8, "Reference text\n(Articles, Trials,\nSEC Filings)", ha="center", va="center", fontsize=8, color="#475569")
    ax.text(1.25, 3.15, "R_src = {s_1, ..., s_M}", ha="center", va="center", fontsize=7.5, family="monospace", color="#1E293B")

    # 2. AI-Generated Text
    box_gen = patches.FancyBboxPatch(
        (0.3, 0.6), 1.9, 1.7, boxstyle="round,pad=0.1,rounding_size=0.15",
        facecolor=c_input, edgecolor=border_c, linewidth=1.2
    )
    ax.add_patch(box_gen)
    ax.text(1.25, 1.95, "Generated Text", ha="center", va="center", fontsize=10, fontweight="bold", color="#0F172A")
    ax.text(1.25, 1.5, "LLM Summary /\nResponse to Audit", ha="center", va="center", fontsize=8, color="#475569")
    ax.text(1.25, 0.85, "T_gen", ha="center", va="center", fontsize=8, family="monospace", color="#1E293B")

    # Stage 1: Atomic Decomposition
    box_s1 = patches.FancyBboxPatch(
        (2.7, 0.6), 2.3, 4.0, boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=c_stage1, edgecolor="#0284C7", linewidth=1.4
    )
    ax.add_patch(box_s1)
    ax.text(3.85, 4.3, "Stage 1", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0369A1")
    ax.text(3.85, 3.95, "Atomic Claim\nDecomposition", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")
    
    # Inner modules
    box_syn = patches.FancyBboxPatch((2.9, 2.5), 1.9, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#38BDF8", linewidth=0.8)
    ax.add_patch(box_syn)
    ax.text(3.85, 3.05, "Syntactic Fallback", ha="center", va="center", fontsize=8, fontweight="bold", color="#0369A1")
    ax.text(3.85, 2.75, "NLTK + Clause Splitter\n(Pronoun Propagation)", ha="center", va="center", fontsize=6.8, color="#334155")

    box_llm = patches.FancyBboxPatch((2.9, 1.3), 1.9, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#38BDF8", linewidth=0.8)
    ax.add_patch(box_llm)
    ax.text(3.85, 1.85, "Few-Shot LLM Mode", ha="center", va="center", fontsize=8, fontweight="bold", color="#0369A1")
    ax.text(3.85, 1.55, "Gemini / OpenAI API\n(Self-Contained Atoms)", ha="center", va="center", fontsize=6.8, color="#334155")

    ax.text(3.85, 0.85, "Output: {c_1, ..., c_K}", ha="center", va="center", fontsize=7.5, family="monospace", fontweight="bold", color="#0369A1")

    # Stage 2: Dense Semantic Retrieval
    box_s2 = patches.FancyBboxPatch(
        (5.5, 0.6), 2.3, 4.0, boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=c_stage2, edgecolor="#7C3AED", linewidth=1.4
    )
    ax.add_patch(box_s2)
    ax.text(6.65, 4.3, "Stage 2", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#6D28D9")
    ax.text(6.65, 3.95, "Dense Semantic\nRetrieval", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    box_bi = patches.FancyBboxPatch((5.7, 2.5), 1.9, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#A78BFA", linewidth=0.8)
    ax.add_patch(box_bi)
    ax.text(6.65, 3.05, "Dense Bi-Encoder", ha="center", va="center", fontsize=8, fontweight="bold", color="#6D28D9")
    ax.text(6.65, 2.75, "all-MiniLM-L6-v2\n(384-dim Embeddings)", ha="center", va="center", fontsize=6.8, color="#334155")

    box_cos = patches.FancyBboxPatch((5.7, 1.3), 1.9, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#A78BFA", linewidth=0.8)
    ax.add_patch(box_cos)
    ax.text(6.65, 1.85, "Cosine Matching", ha="center", va="center", fontsize=8, fontweight="bold", color="#6D28D9")
    ax.text(6.65, 1.55, "Top-2 Source Sentences\nw_i = cos(e_s, e_c)", ha="center", va="center", fontsize=6.8, color="#334155")

    ax.text(6.65, 0.85, "Output: (premise, claim)", ha="center", va="center", fontsize=7.5, family="monospace", fontweight="bold", color="#6D28D9")

    # Stage 3: Cross-Encoder NLI Verification
    box_s3 = patches.FancyBboxPatch(
        (8.3, 0.6), 2.4, 4.0, boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=c_stage3, edgecolor="#D97706", linewidth=1.4
    )
    ax.add_patch(box_s3)
    ax.text(9.5, 4.3, "Stage 3", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#B45309")
    ax.text(9.5, 3.95, "Cross-Encoder NLI\nVerification", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    box_deberta = patches.FancyBboxPatch((8.5, 2.5), 2.0, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#FBBF24", linewidth=0.8)
    ax.add_patch(box_deberta)
    ax.text(9.5, 3.05, "DeBERTa-v3 Cross-Encoder", ha="center", va="center", fontsize=8, fontweight="bold", color="#B45309")
    ax.text(9.5, 2.75, "MNLI + FEVER + ANLI\nSoftmax Logits [ent, neu, con]", ha="center", va="center", fontsize=6.8, color="#334155")

    box_thresh = patches.FancyBboxPatch((8.5, 1.3), 2.0, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#FBBF24", linewidth=0.8)
    ax.add_patch(box_thresh)
    ax.text(9.5, 1.85, "Decision Thresholding", ha="center", va="center", fontsize=8, fontweight="bold", color="#B45309")
    ax.text(9.5, 1.55, "P_ent >= 0.55 -> SUP\nP_con >= 0.45 -> CON", ha="center", va="center", fontsize=6.8, color="#334155")

    ax.text(9.5, 0.85, "Verdicts: {v_1, ..., v_K}", ha="center", va="center", fontsize=7.5, family="monospace", fontweight="bold", color="#B45309")

    # Stage 4: Trust Score Calibration & Audit Report
    box_s4 = patches.FancyBboxPatch(
        (11.2, 0.6), 2.5, 4.0, boxstyle="round,pad=0.12,rounding_size=0.18",
        facecolor=c_stage4, edgecolor="#059669", linewidth=1.4
    )
    ax.add_patch(box_s4)
    ax.text(12.45, 4.3, "Stage 4", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#047857")
    ax.text(12.45, 3.95, "Grounding Trust\nCalibration", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    box_gts = patches.FancyBboxPatch((11.45, 2.5), 2.0, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#34D399", linewidth=0.8)
    ax.add_patch(box_gts)
    ax.text(12.45, 3.05, "Non-Linear Trust Score", ha="center", va="center", fontsize=8, fontweight="bold", color="#047857")
    ax.text(12.45, 2.75, "GTS = 100 * sum(w_i P_ent)\n- Omega(K) Penalty", ha="center", va="center", fontsize=6.8, color="#334155")

    box_rep = patches.FancyBboxPatch((11.45, 1.3), 2.0, 0.9, boxstyle="round,pad=0.05,rounding_size=0.08", facecolor="#FFFFFF", edgecolor="#34D399", linewidth=0.8)
    ax.add_patch(box_rep)
    ax.text(12.45, 1.85, "Audit Dashboard & Reports", ha="center", va="center", fontsize=8, fontweight="bold", color="#047857")
    ax.text(12.45, 1.55, "Streamlit UI • audit.json\nAnnotated Text Highlights", ha="center", va="center", fontsize=6.8, color="#334155")

    ax.text(12.45, 0.85, "GTS in [0, 100%]", ha="center", va="center", fontsize=8, family="monospace", fontweight="bold", color="#047857")

    # Connective Arrows
    arrow_props = dict(arrowstyle="->,head_width=0.35,head_length=0.45", lw=1.3, color="#334155")

    # Gen -> Stage 1
    ax.annotate("", xy=(2.7, 1.45), xytext=(2.2, 1.45), arrowprops=arrow_props)
    
    # Stage 1 -> Stage 2 (Claims)
    ax.annotate("", xy=(5.5, 2.6), xytext=(5.0, 2.6), arrowprops=arrow_props)
    
    # Source -> Stage 2 (Source Sentences)
    ax.annotate("", xy=(5.5, 3.75), xytext=(2.2, 3.75), arrowprops=arrow_props)
    
    # Stage 2 -> Stage 3 (Evidence + Claim)
    ax.annotate("", xy=(8.3, 2.6), xytext=(7.8, 2.6), arrowprops=arrow_props)
    
    # Stage 3 -> Stage 4 (Verdicts + Probs)
    ax.annotate("", xy=(11.2, 2.6), xytext=(10.7, 2.6), arrowprops=arrow_props)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_fig2_ablation_comparison(metrics_path: Path, output_path: Path) -> None:
    """Generate grouped bar chart comparing Baseline vs FactLens across primary metrics."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    b_sum = data["baseline_whole_document"]
    f_sum = data["factlens_atomic_claim"]

    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1-Score"]
    baseline_scores = [
        b_sum["accuracy"] * 100,
        b_sum["precision_macro"] * 100,
        b_sum["recall_macro"] * 100,
        b_sum["f1_macro"] * 100,
    ]
    factlens_scores = [
        f_sum["accuracy"] * 100,
        f_sum["precision_macro"] * 100,
        f_sum["recall_macro"] * 100,
        f_sum["f1_macro"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    
    x = np.arange(len(metrics))
    width = 0.35

    color_base = "#64748B"    # Slate Gray
    color_fact = "#2563EB"    # Academic Royal Blue

    rects1 = ax.bar(x - width/2, baseline_scores, width, label="Baseline Whole-Document NLI", color=color_base, edgecolor="#334155", linewidth=1.0)
    rects2 = ax.bar(x + width/2, factlens_scores, width, label="FactLens Atomic Claim Pipeline", color=color_fact, edgecolor="#1E40AF", linewidth=1.0)

    # Add text labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#334155")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1E40AF")

    # Add Delta improvement callouts
    deltas = [f - b for f, b in zip(factlens_scores, baseline_scores)]
    for idx, (xi, d) in enumerate(zip(x, deltas)):
        ax.annotate(f"+{d:.1f}%",
                    xy=(xi + width/2, factlens_scores[idx] / 2),
                    ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#10B981", edgecolor="none", alpha=0.9))

    ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title("Primary Classification Performance: Baseline vs. FactLens Pipeline (N = 30)",
                 fontsize=12, fontweight="bold", pad=14, color="#0F172A")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5, fontweight="bold", color="#1E293B")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)

    # Clean legend
    ax.legend(loc="upper left", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)
    
    # Subtle citation footnote
    ax.text(0.99, 0.02, "Evaluated on multi-domain benchmark dataset (Medical, Financial, Indian Policy, Technical)",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom", color="#64748B", style="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_fig3_category_breakdown(metrics_path: Path, output_path: Path) -> None:
    """Generate category-wise F1 comparison across the 4 hallucination error categories."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_data = data["category_breakdown"]
    
    categories = [
        "Numerical Distortion",
        "Extrapolated Fabrication",
        "Fully Grounded",
        "Entity/Attribute Swap",
    ]

    base_f1 = [cat_data[c]["baseline_f1"] * 100 for c in categories]
    fact_f1 = [cat_data[c]["factlens_f1"] * 100 for c in categories]
    sample_counts = [cat_data[c]["sample_count"] for c in categories]

    # Category labels with sample count
    labels = [f"{c}\n(n = {s})" for c, s in zip(categories, sample_counts)]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)

    y = np.arange(len(categories))
    height = 0.35

    color_base = "#94A3B8"    # Neutral Slate
    color_fact = "#0D9488"    # Academic Teal / Deep Emerald

    rects1 = ax.barh(y + height/2, base_f1, height, label="Baseline Whole-Doc NLI F1", color=color_base, edgecolor="#475569")
    rects2 = ax.barh(y - height/2, fact_f1, height, label="FactLens Atomic Claim NLI F1", color=color_fact, edgecolor="#0F766E")

    # Add data labels
    for rect in rects1:
        w = rect.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold", color="#475569")

    for rect in rects2:
        w = rect.get_width()
        ax.annotate(f"{w:.1f}%",
                    xy=(w, rect.get_y() + rect.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="bold", color="#0F766E")

    # Add Delta tags
    for idx, (b, f) in enumerate(zip(base_f1, fact_f1)):
        delta = f - b
        delta_color = "#047857" if delta >= 30 else "#D97706"
        ax.annotate(f"Delta: +{delta:.1f}%",
                    xy=(103, y[idx] - height/2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, fontweight="bold", color=delta_color)

    ax.set_xlabel("Macro F1-Score (%)", fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_title("Hallucination Category Breakdown: Baseline vs. FactLens F1-Score",
                 fontsize=12, fontweight="bold", pad=14, color="#0F172A")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.invert_yaxis()  # top category first
    ax.set_xlim(0, 122)
    ax.grid(axis="x", linestyle="--", alpha=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    figures_dir = ensure_output_dir(base_dir / "report" / "figures")
    metrics_path = base_dir / "results" / "benchmark_metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Please generate benchmark results first.")

    print(f"Generating figures into {figures_dir}...")
    generate_fig1_pipeline_architecture(figures_dir / "fig1_pipeline_architecture.png")
    generate_fig2_ablation_comparison(metrics_path, figures_dir / "fig2_ablation_comparison.png")
    generate_fig3_category_breakdown(metrics_path, figures_dir / "fig3_category_breakdown.png")
    print("All 3 figures successfully generated at 300 DPI!")


if __name__ == "__main__":
    main()
