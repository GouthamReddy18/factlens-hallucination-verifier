"""Benchmark and Ablation Evaluation Module for FactLens.

Compares FactLens Atomic Claim-Level NLI against a Naive Whole-Text NLI Baseline
across 30 curated benchmark cases covering Fully Grounded statements, Numerical Distortion,
Entity/Attribute Swap, and Extrapolated Fabrication.
"""

from __future__ import annotations

import json
import time
import os
import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.decomposer import ClaimDecomposer
from src.retriever import EvidenceRetriever
from src.verifier import GroundingVerifier

logger = logging.getLogger(__name__)


class FactLensEvaluator:
    """Evaluates grounding and hallucination verification performance."""

    def __init__(
        self,
        dataset_path: str = "data/benchmark_dataset.json",
        results_dir: str = "results",
        device: Optional[str] = None,
        retriever: Optional[EvidenceRetriever] = None,
        verifier: Optional[GroundingVerifier] = None,
    ) -> None:
        """Initialize evaluator with benchmark dataset path and optional shared pipeline instances."""
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.decomposer = ClaimDecomposer(default_mode="syntactic")
        self.retriever = retriever if retriever is not None else EvidenceRetriever(device=self.device, lazy_load=True)
        self.verifier = verifier if verifier is not None else GroundingVerifier(device=self.device, lazy_load=True)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(
        self,
        limit: Optional[int] = None,
        sample_per_category: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute full evaluation across both Baseline and FactLens pipeline.

        Args:
            limit: Maximum total test cases to evaluate (first N).
            sample_per_category: If provided, take N balanced samples per category.
        """
        raw_dataset = self.load_dataset()

        if sample_per_category is not None and sample_per_category > 0:
            dataset: List[Dict[str, Any]] = []
            seen_counts: Dict[str, int] = {}
            for item in raw_dataset:
                cat = item["category"]
                seen_counts[cat] = seen_counts.get(cat, 0)
                if seen_counts[cat] < sample_per_category:
                    dataset.append(item)
                    seen_counts[cat] += 1
        elif limit is not None:
            dataset = raw_dataset[:limit]
        else:
            dataset = raw_dataset

        logger.info(f"Starting evaluation on {len(dataset)} test cases...")

        y_true: List[str] = []
        y_pred_baseline: List[str] = []
        y_pred_factlens: List[str] = []

        baseline_latencies: List[float] = []
        factlens_latencies: List[float] = []

        detailed_results: List[Dict[str, Any]] = []

        for idx, item in enumerate(dataset, 1):
            tc_id = item["id"]
            category = item["category"]
            domain = item["domain"]
            source_text = item["source_text"]
            generated_text = item["generated_text"]
            ground_truth = item["ground_truth_label"]  # 'GROUNDED' or 'HALLUCINATED'

            y_true.append(ground_truth)

            # ----------------------------------------------------
            # 1. Baseline: Whole-Document Cross-Encoder NLI
            # ----------------------------------------------------
            t0_b = time.perf_counter()
            base_out = self.verifier.verify_whole_document(source_text, generated_text)
            t_base = (time.perf_counter() - t0_b) * 1000.0
            baseline_latencies.append(t_base)

            # In whole-doc NLI, if output is SUPPORTED -> GROUNDED, else HALLUCINATED
            pred_b = "GROUNDED" if base_out["verdict"] == "SUPPORTED" else "HALLUCINATED"
            y_pred_baseline.append(pred_b)

            # ----------------------------------------------------
            # 2. FactLens: Decompose -> Retrieve -> Verify
            # ----------------------------------------------------
            t0_f = time.perf_counter()
            claims = self.decomposer.decompose(generated_text, mode="syntactic")
            retrieval_res = self.retriever.retrieve_evidence(source_text, claims, top_k=2)
            factlens_out = self.verifier.verify_pipeline(
                retrieval_results=retrieval_res,
                run_baseline=False,
            )
            t_factlens = (time.perf_counter() - t0_f) * 1000.0
            factlens_latencies.append(t_factlens)

            # FactLens classification rule:
            # If any contradiction detected or trust score < 65% -> HALLUCINATED
            # If 0 contradictions and trust score >= 65% -> GROUNDED
            if factlens_out.counts["contradicted"] > 0 or factlens_out.trust_score < 65.0:
                pred_f = "HALLUCINATED"
            else:
                pred_f = "GROUNDED"
            y_pred_factlens.append(pred_f)

            detailed_results.append({
                "id": tc_id,
                "category": category,
                "domain": domain,
                "ground_truth": ground_truth,
                "baseline_pred": pred_b,
                "baseline_verdict": base_out["verdict"],
                "baseline_latency_ms": round(t_base, 2),
                "factlens_pred": pred_f,
                "factlens_trust_score": factlens_out.trust_score,
                "factlens_claims_count": len(claims),
                "factlens_counts": factlens_out.counts,
                "factlens_latency_ms": round(t_factlens, 2),
            })

            logger.info(f"[{idx}/{len(dataset)}] Case {tc_id} ({category}): Truth={ground_truth} | Base={pred_b} | FactLens={pred_f}")

            # Incremental checkpoint to results/benchmark_metrics.json
            metrics = self._compute_metrics(
                y_true=y_true,
                y_pred_baseline=y_pred_baseline,
                y_pred_factlens=y_pred_factlens,
                baseline_latencies=baseline_latencies,
                factlens_latencies=factlens_latencies,
                detailed_results=detailed_results,
            )
            output_file = self.results_dir / "benchmark_metrics.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        logger.info(f"Benchmark metrics successfully saved to {output_file}")
        return metrics

    def _compute_metrics(
        self,
        y_true: List[str],
        y_pred_baseline: List[str],
        y_pred_factlens: List[str],
        baseline_latencies: List[float],
        factlens_latencies: List[float],
        detailed_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute macro classification metrics and category breakdown."""
        labels = ["GROUNDED", "HALLUCINATED"]

        def calc_summary(y_pred: List[str], latencies: List[float]) -> Dict[str, float]:
            return {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
                "avg_latency_ms": round(float(sum(latencies) / max(len(latencies), 1)), 2),
            }

        baseline_summary = calc_summary(y_pred_baseline, baseline_latencies)
        factlens_summary = calc_summary(y_pred_factlens, factlens_latencies)

        # Category-wise analysis
        categories = sorted(list(set(d["category"] for d in detailed_results)))
        category_metrics: Dict[str, Dict[str, Any]] = {}

        for cat in categories:
            cat_items = [d for d in detailed_results if d["category"] == cat]
            sub_y_true = [d["ground_truth"] for d in cat_items]
            sub_y_base = [d["baseline_pred"] for d in cat_items]
            sub_y_fact = [d["factlens_pred"] for d in cat_items]

            category_metrics[cat] = {
                "sample_count": len(cat_items),
                "baseline_accuracy": round(float(accuracy_score(sub_y_true, sub_y_base)), 4),
                "factlens_accuracy": round(float(accuracy_score(sub_y_true, sub_y_fact)), 4),
                "baseline_f1": round(float(f1_score(sub_y_true, sub_y_base, average="macro", zero_division=0)), 4),
                "factlens_f1": round(float(f1_score(sub_y_true, sub_y_fact, average="macro", zero_division=0)), 4),
            }

        return {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(y_true),
            "baseline_whole_document": baseline_summary,
            "factlens_atomic_claim": factlens_summary,
            "category_breakdown": category_metrics,
            "detailed_results": detailed_results,
        }

    def print_markdown_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics into a clean markdown table for console/README."""
        base = metrics["baseline_whole_document"]
        fact = metrics["factlens_atomic_claim"]
        cats = metrics["category_breakdown"]

        md = []
        md.append("### FactLens vs. Baseline Whole-Document Evaluation Results\n")
        md.append("| Metric | Baseline (Whole-Document NLI) | FactLens (Atomic Claim NLI) | Relative Improvement |")
        md.append("|---|---|---|---|")

        acc_diff = (fact['accuracy'] - base['accuracy']) * 100
        f1_diff = (fact['f1_macro'] - base['f1_macro']) * 100
        prec_diff = (fact['precision_macro'] - base['precision_macro']) * 100
        rec_diff = (fact['recall_macro'] - base['recall_macro']) * 100

        md.append(f"| **Accuracy** | {base['accuracy']*100:.1f}% | **{fact['accuracy']*100:.1f}%** | `+{acc_diff:.1f}%` |")
        md.append(f"| **Macro Precision** | {base['precision_macro']*100:.1f}% | **{fact['precision_macro']*100:.1f}%** | `+{prec_diff:.1f}%` |")
        md.append(f"| **Macro Recall** | {base['recall_macro']*100:.1f}% | **{fact['recall_macro']*100:.1f}%** | `+{rec_diff:.1f}%` |")
        md.append(f"| **Macro F1-Score** | {base['f1_macro']*100:.1f}% | **{fact['f1_macro']*100:.1f}%** | `+{f1_diff:.1f}%` |")
        md.append(f"| **Avg Latency (ms)** | {base['avg_latency_ms']:.1f} ms | {fact['avg_latency_ms']:.1f} ms | Sub-second real-time |")

        md.append("\n#### Category-wise F1-Score Breakdown\n")
        md.append("| Category | Samples | Baseline F1 | FactLens F1 | Delta |")
        md.append("|---|---|---|---|---|")
        for cat_name, c_data in cats.items():
            delta = (c_data['factlens_f1'] - c_data['baseline_f1']) * 100
            md.append(f"| **{cat_name}** | {c_data['sample_count']} | {c_data['baseline_f1']*100:.1f}% | **{c_data['factlens_f1']*100:.1f}%** | `+{delta:.1f}%` |")

        md_str = "\n".join(md)
        print("\n" + md_str + "\n")
        return md_str


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run FactLens Grounding & Hallucination Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N cases")
    parser.add_argument("--sample-per-category", "-s", type=int, default=None, help="Evaluate balanced N cases per category")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluator = FactLensEvaluator(device=args.device)
    results = evaluator.run_benchmark(limit=args.limit, sample_per_category=args.sample_per_category)
    evaluator.print_markdown_table(results)
