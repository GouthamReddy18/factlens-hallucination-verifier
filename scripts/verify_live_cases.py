"""FactLens: Standalone live verification script for Aditya-L1 and RBI cases."""

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.decomposer import ClaimDecomposer
from src.retriever import EvidenceRetriever
from src.verifier import GroundingVerifier


def main():
    print("=" * 70)
    print("FactLens Live Verification: Aditya-L1 & RBI Monetary Policy Cases")
    print("=" * 70)

    decomposer = ClaimDecomposer(default_mode="syntactic")
    retriever = EvidenceRetriever(device="cpu", lazy_load=False)
    verifier = GroundingVerifier(device="cpu", lazy_load=False)

    # ---------------------------------------------------------
    # TEST 1: Aditya-L1 Supported Text (4 claims)
    # ---------------------------------------------------------
    print("\n[TEST 1] Aditya-L1 Supported Case:")
    src_aditya = (
        "Aditya-L1 is the first Indian space-based observatory designed to study the Sun. "
        "The spacecraft was successfully placed in a halo orbit around the Sun-Earth Lagrange point L1, "
        "located about 1.5 million kilometers from Earth. "
        "The observatory carries seven scientific payloads developed to observe the solar photosphere, "
        "chromosphere, and corona. "
        "The mission was launched on September 2, 2023, aboard the PSLV-C57 rocket from Sriharikota."
    )
    gen_aditya = (
        "Aditya-L1 is the first Indian space-based observatory designed to study the Sun. "
        "The spacecraft was placed in a halo orbit around Lagrange point L1 located about 1.5 million kilometers from Earth. "
        "The observatory carries seven scientific payloads developed to observe the solar photosphere, chromosphere. "
        "The mission was launched on September 2, 2023, aboard the PSLV-C57 rocket from Sriharikota."
    )

    claims_aditya = decomposer.decompose(gen_aditya)
    ret_aditya = retriever.retrieve_evidence(src_aditya, claims_aditya, top_k=2)
    res_aditya = verifier.verify_pipeline(ret_aditya)

    print(f"Overall Trust Score: {res_aditya.trust_score:.1f}% | Verdict: {res_aditya.verdict}")
    for i, c in enumerate(res_aditya.claims, 1):
        ent = c.probabilities.get("entailment", 0.0)
        print(f"  Claim {i}: {c.claim}")
        print(f"    Verdict: {c.verdict} | Entailment: {ent:.1%} | Neutral: {c.probabilities.get('neutral', 0.0):.1%} | Contra: {c.probabilities.get('contradiction', 0.0):.1%}")
        print(f"    Selected Evidence: {c.evidence_text}")
        assert ent > 0.90, f"Aditya Claim {i} failed: entailment {ent:.2%} <= 90%"
        assert c.verdict == "SUPPORTED", f"Aditya Claim {i} verdict was {c.verdict}, expected SUPPORTED"

    print(">>> PASS: All 4 Aditya-L1 claims scored >90% Entailment (SUPPORTED)!")

    # ---------------------------------------------------------
    # TEST 2: RBI Monetary Policy Contradicted Case (3.1% vs 7.2%)
    # ---------------------------------------------------------
    print("\n[TEST 2] RBI Monetary Policy Case (Growth Distortion 3.1% vs 7.2%):")
    src_rbi = (
        "The Reserve Bank of India kept the repo rate unchanged at 6.50 percent. "
        "Economic growth was estimated at 7.2 percent for the fiscal year. "
        "The monsoon season showed above normal rainfall."
    )
    gen_rbi = (
        "The RBI maintained the benchmark repo rate at 6.50 percent. "
        "Economic growth was estimated at 3.1 percent for the fiscal year. "
        "Foreign direct investment surged by 25 percent."
    )

    claims_rbi = decomposer.decompose(gen_rbi)
    ret_rbi = retriever.retrieve_evidence(src_rbi, claims_rbi, top_k=2)
    res_rbi = verifier.verify_pipeline(ret_rbi)

    print(f"Overall Trust Score: {res_rbi.trust_score:.1f}% | Verdict: {res_rbi.verdict}")
    for i, c in enumerate(res_rbi.claims, 1):
        print(f"  Claim {i}: {c.claim}")
        print(f"    Verdict: {c.verdict} | Entailment: {c.probabilities.get('entailment', 0.0):.1%} | Neutral: {c.probabilities.get('neutral', 0.0):.1%} | Contra: {c.probabilities.get('contradiction', 0.0):.1%}")
        print(f"    Selected Evidence: {c.evidence_text}")

    # Check Claim 1: Supported (>55%)
    assert res_rbi.claims[0].verdict == "SUPPORTED"
    # Check Claim 2: Contradicted (>90%)
    contra_growth = res_rbi.claims[1].probabilities.get("contradiction", 0.0)
    assert contra_growth > 0.90, f"RBI Claim 2 contradiction {contra_growth:.2%} <= 90%"
    assert res_rbi.claims[1].verdict == "CONTRADICTED"
    # Check Claim 3: Unverified
    assert res_rbi.claims[2].verdict == "UNVERIFIED"

    print(">>> PASS: RBI growth distortion scored >90% Contradiction (CONTRADICTED)!")
    print("\n" + "=" * 70)
    print("ALL LIVE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
