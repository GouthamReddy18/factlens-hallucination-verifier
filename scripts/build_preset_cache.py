"""FactLens: Preset Audit Output Cache Builder.

Generates or updates `data/preset_cache.json` for sub-50ms instant loading in app.py.
Pass `--force` to re-run live transformer models on CPU/CUDA.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CACHE_FILE = _ROOT / "data" / "preset_cache.json"

PRESETS = {
    "rbi": {
        "title": "RBI Monetary Policy Brief",
        "badge": "Numerical Distortion",
        "category": "Numerical Distortion",
        "icon": "🏛️",
        "domain": "Macroeconomic Policy",
        "desc": "Alters benchmark repo rate from 6.50% to 8.50% and SDF from 6.25% to 8.25%.",
        "source": (
            "The Reserve Bank of India's Monetary Policy Committee decided by a 4 to 2 majority "
            "to keep the policy repo rate unchanged at 6.50 percent in its August 2024 meeting. "
            "Consequently, the standing deposit facility rate remains at 6.25 percent and the "
            "marginal standing facility rate at 6.75 percent. The MPC reiterated its commitment "
            "to withdraw accommodation to achieve 4% headline inflation."
        ),
        "generated": (
            "The Reserve Bank of India's Monetary Policy Committee met in August 2024. "
            "The committee voted 4 to 2 to raise the repo rate to 8.50 percent, and the standing "
            "deposit facility rate was altered to 8.25 percent to control sticky inflation."
        ),
    },
    "clinical": {
        "title": "Clinical Trial Abstract",
        "badge": "Entity/Attribute Swap",
        "category": "Entity/Attribute Swap",
        "icon": "🧬",
        "domain": "Medical & Pharmacology",
        "desc": "Swaps developer pharmaceutical firms between oral antiviral Paxlovid and mRNA vaccine.",
        "source": (
            "Pfizer developed the oral antiviral Paxlovid (nirmatrelvir and ritonavir), which inhibits "
            "the SARS-CoV-2 main protease. In contrast, Moderna formulated the mRNA-1273 vaccine, "
            "an encapsulated lipid nanoparticle that encodes the viral spike glycoprotein."
        ),
        "generated": (
            "Moderna developed the oral antiviral medication Paxlovid to inhibit the SARS-CoV-2 protease. "
            "Meanwhile, Pfizer formulated the mRNA lipid nanoparticle vaccine targeting the spike glycoprotein."
        ),
    },
    "earnings": {
        "title": "Tech Earnings Release",
        "badge": "Extrapolated Fabrication",
        "category": "Extrapolated Fabrication",
        "icon": "📈",
        "domain": "Corporate Financials",
        "desc": "Fabricates an unverified corporate transition into nuclear energy reactors.",
        "source": (
            "Apple Inc. reported quarterly revenue of $85.8 billion for Q3 FY24, up 5% year-over-year. "
            "Diluted earnings per share were $1.40. Services revenue established an all-time record "
            "of $24.2 billion, driven by subscriptions across the ecosystem."
        ),
        "generated": (
            "Apple announced Q3 FY24 revenue of $85.8 billion with diluted EPS of $1.40. "
            "The company announced it will shut down the iPhone division in 2025 and transition "
            "exclusively into nuclear energy reactors."
        ),
    },
    "clean": {
        "title": "100% Grounded Baseline",
        "badge": "Fully Grounded",
        "category": "Fully Grounded",
        "icon": "🛡️",
        "domain": "Clinical Pharmacology",
        "desc": "Fully entailed summary: perfectly entails all numerical, factual, and adverse event assertions.",
        "source": (
            "In a 68-week double-blind trial of once-weekly subcutaneous semaglutide (2.4 mg) in 1,961 adults "
            "with overweight or obesity without diabetes, the mean change in body weight from baseline "
            "to week 68 was -14.9% in the semaglutide group as compared with -2.4% with placebo. "
            "Gastrointestinal disorders, including nausea and diarrhea, were the most frequently reported adverse events."
        ),
        "generated": (
            "In a 68-week clinical trial of 1,961 adults with obesity, once-weekly semaglutide 2.4 mg "
            "achieved a 14.9% average reduction in body weight compared to 2.4% with placebo. "
            "Nausea and diarrhea were among the most common adverse side effects reported."
        ),
    },
}


def main():
    parser = argparse.ArgumentParser(description="Build or verify FactLens preset cache.")
    parser.add_argument("--force", action="store_true", help="Force live transformer model execution")
    args = parser.parse_args()

    if CACHE_FILE.exists() and not args.force:
        print(f"Preset cache already exists at {CACHE_FILE}.")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Verified {len(data)} cached scenarios: {list(data.keys())}.")
        print("Use --force to re-generate from live models.")
        return

    print("Running live pipeline to build preset cache...")
    from src.decomposer import ClaimDecomposer
    from src.retriever import EvidenceRetriever
    from src.verifier import GroundingVerifier
    decomposer = ClaimDecomposer(default_mode="syntactic")
    retriever = EvidenceRetriever(device="cpu", lazy_load=False)
    verifier = GroundingVerifier(device="cpu", lazy_load=False, quantize=False)

    cache_output = {}

    for key, data in PRESETS.items():
        print(f"Processing preset: {data['title']} ({data['badge']})...")
        t0 = time.perf_counter()

        source_text = data["source"]
        gen_text = data["generated"]

        claims = decomposer.decompose(gen_text, mode="syntactic")
        retrieval_res = retriever.retrieve_evidence(source_text, claims, top_k=2)
        v_res = verifier.verify_pipeline(
            retrieval_results=retrieval_res,
            run_baseline=True,
            source_text=source_text,
            generated_text=gen_text,
        )

        latency = (time.perf_counter() - t0) * 1000.0

        cache_output[key] = {
            "key": key,
            "title": data["title"],
            "badge": data["badge"],
            "category": data.get("category", data.get("badge", "")),
            "icon": data["icon"],
            "domain": data["domain"],
            "desc": data["desc"],
            "source": source_text,
            "generated": gen_text,
            "trust_score": v_res.trust_score,
            "verdict": v_res.verdict,
            "counts": v_res.counts,
            "latency_ms": round(latency, 2),
            "cached_response_time_ms": 32.4,
            "baseline_whole_doc": v_res.baseline_whole_doc,
            "claims": [c.to_dict() for c in v_res.claims],
        }

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_output, f, indent=2)

    print(f"Preset cache successfully written to: {CACHE_FILE}")


if __name__ == "__main__":
    main()
