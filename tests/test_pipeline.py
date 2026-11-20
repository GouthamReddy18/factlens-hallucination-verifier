"""Unit and integration tests for the FactLens verification pipeline.

Tests Stage 1 (Decomposition), Stage 2 (Retrieval), Stage 3 (NLI Verification),
and Stage 4 (Benchmark Evaluation).
"""

import pytest
import os
import json
from pathlib import Path

from src.decomposer import ClaimDecomposer
from src.retriever import EvidenceRetriever
from src.verifier import GroundingVerifier, ClaimVerification, VerificationResult
from src.evaluator import FactLensEvaluator


# =========================================================================
# STAGE 1: CLAIM DECOMPOSITION TESTS
# =========================================================================

def test_decomposer_syntactic_basic():
    decomposer = ClaimDecomposer(default_mode="syntactic")
    text = "The company reported revenue of $10 billion and expanded into 5 new countries."
    claims = decomposer.decompose(text, mode="syntactic")

    assert len(claims) >= 1
    assert any("revenue" in c.lower() for c in claims)
    for c in claims:
        assert c.endswith((".", "!", "?"))
        assert len(c.split()) >= 3


def test_decomposer_compound_clauses():
    decomposer = ClaimDecomposer(default_mode="syntactic")
    text = (
        "The Reserve Bank kept repo rates at 6.50%; however, inflation remained elevated. "
        "The committee voted 4 to 2 in favor of the pause."
    )
    claims = decomposer.decompose(text, mode="syntactic")

    assert len(claims) >= 2
    # Verify semicolon or sentence separation
    assert any("6.50" in c for c in claims)
    assert any("4 to 2" in c for c in claims)


def test_decomposer_empty_input():
    decomposer = ClaimDecomposer(default_mode="syntactic")
    assert decomposer.decompose("") == []
    assert decomposer.decompose("   ") == []


def test_decomposer_fallback_on_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    decomposer = ClaimDecomposer(default_mode="llm", api_key=None)
    # Should safely fallback to syntactic without crashing
    claims = decomposer.decompose("Apple announced quarterly revenue of $85 billion.", mode="llm")
    assert len(claims) >= 1
    assert "85 billion" in claims[0]


# =========================================================================
# STAGE 2: DENSE EVIDENCE RETRIEVAL TESTS
# =========================================================================

@pytest.fixture(scope="module")
def retriever():
    return EvidenceRetriever(device="cpu", lazy_load=False)


def test_retriever_sentence_splitting(retriever):
    source = (
        "In a phase 3 clinical trial, 43,448 participants were enrolled. "
        "The vaccine demonstrated 95% efficacy against symptomatic infection. "
        "Side effects included headache and fatigue."
    )
    sentences = retriever.split_sentences(source)
    assert len(sentences) == 3
    assert "43,448" in sentences[0]
    assert "95%" in sentences[1]


def test_retriever_top_k_similarity(retriever):
    source = (
        "The Reserve Bank of India kept the repo rate unchanged at 6.50 percent. "
        "Economic growth was estimated at 7.2 percent for the fiscal year. "
        "The monsoon season showed above normal rainfall."
    )
    claims = [
        "The RBI maintained the benchmark repo rate at 6.50 percent.",
        "Monsoon precipitation was above average.",
    ]

    results = retriever.retrieve_evidence(source, claims, top_k=2)
    assert len(results) == 2

    # Check first claim retrieval
    res_0 = results[0]
    assert res_0["claim"] == claims[0]
    assert len(res_0["evidence_sentences"]) == 2
    assert "repo rate" in res_0["evidence_sentences"][0]
    assert res_0["top_score"] > 0.60  # High semantic similarity expected


def test_retriever_empty_claims(retriever):
    assert retriever.retrieve_evidence("Some source text.", []) == []


def test_retriever_caching(retriever):
    source = "The Federal Reserve raised rates by 25 basis points in the latest session."
    retriever.retrieve_evidence(source, ["The Fed raised interest rates."])
    key = hash(source)
    assert key in retriever._sentence_cache, "Source sentence embeddings must be cached in memory"


# =========================================================================
# STAGE 3: CROSS-ENCODER NLI VERIFICATION TESTS
# =========================================================================

@pytest.fixture(scope="module")
def verifier():
    return GroundingVerifier(device="cpu", lazy_load=False)


def test_verifier_supported_pair(verifier):
    premise = "Apple posted quarterly revenue of $85.8 billion, an increase of 5 percent year over year."
    hypothesis = "Apple achieved quarterly revenue of $85.8 billion."

    res = verifier.verify_pair(premise=premise, hypothesis=hypothesis, evidence_similarity=0.88)
    assert res.verdict == "SUPPORTED"
    assert res.probabilities["entailment"] >= 0.55
    assert res.probabilities["entailment"] > res.probabilities["contradiction"]


def test_verifier_contradicted_pair(verifier):
    premise = "The repo rate was kept unchanged at 6.50 percent."
    hypothesis = "The repo rate was drastically increased to 12.50 percent."

    res = verifier.verify_pair(premise=premise, hypothesis=hypothesis, evidence_similarity=0.75)
    assert res.verdict == "CONTRADICTED"
    assert res.probabilities["contradiction"] >= 0.45
    assert res.probabilities["contradiction"] > res.probabilities["entailment"]


def test_verifier_low_similarity_unverified(verifier):
    premise = "The moon orbits the Earth at an average distance of 384,400 kilometers."
    hypothesis = "The central bank lowered the benchmark interest rate by 50 basis points."

    res = verifier.verify_pair(premise=premise, hypothesis=hypothesis, evidence_similarity=0.15)
    assert res.verdict == "UNVERIFIED"


def test_verifier_trust_score_calculation(verifier):
    # Case with 2 supported claims -> High trust score
    supported_claims = [
        ClaimVerification(
            claim="Claim 1",
            evidence_text="Evidence 1",
            evidence_similarity=0.8,
            verdict="SUPPORTED",
            probabilities={"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01},
            explanation="Supported",
        ),
        ClaimVerification(
            claim="Claim 2",
            evidence_text="Evidence 2",
            evidence_similarity=0.85,
            verdict="SUPPORTED",
            probabilities={"entailment": 0.90, "neutral": 0.08, "contradiction": 0.02},
            explanation="Supported",
        ),
    ]
    high_score = verifier._compute_trust_score(supported_claims)
    assert high_score >= 85.0

    # Case with contradiction -> Penalized score
    mixed_claims = [
        supported_claims[0],
        ClaimVerification(
            claim="Claim 3",
            evidence_text="Evidence 3",
            evidence_similarity=0.8,
            verdict="CONTRADICTED",
            probabilities={"entailment": 0.02, "neutral": 0.08, "contradiction": 0.90},
            explanation="Contradicted",
        ),
    ]
    low_score = verifier._compute_trust_score(mixed_claims)
    assert low_score < high_score
    assert low_score <= 40.0


def test_verifier_whole_document_baseline(verifier):
    source = "The company's net income for 2023 was $5 billion."
    generated = "The company reported a net income of $5 billion in 2023."

    baseline = verifier.verify_whole_document(source, generated)
    assert "verdict" in baseline
    assert "probabilities" in baseline
    assert baseline["verdict"] == "SUPPORTED"


def test_verifier_cpu_optimization_and_recompute(verifier):
    assert verifier.device == "cpu"
    # Verify decoupled zero-lag recomputation works instantaneously
    sample_claims = [
        {
            "claim": "Apple revenue was $85.8 billion.",
            "evidence_text": "Apple posted revenue of $85.8 billion.",
            "evidence_similarity": 0.88,
            "probabilities": {"entailment": 0.92, "neutral": 0.06, "contradiction": 0.02},
        },
        {
            "claim": "The repo rate was drastically increased to 12.50 percent.",
            "evidence_text": "The repo rate was kept unchanged at 6.50 percent.",
            "evidence_similarity": 0.78,
            "probabilities": {"entailment": 0.03, "neutral": 0.10, "contradiction": 0.87},
        },
    ]

    # Default thresholds -> 1 supported, 1 contradicted -> HALLUCINATED
    res_default = verifier.recompute_verification_result(
        sample_claims,
        entailment_thresh=0.55,
        contradiction_thresh=0.45,
        min_similarity_thresh=0.25,
    )
    assert res_default.counts["supported"] == 1
    assert res_default.counts["contradicted"] == 1
    assert res_default.verdict == "HALLUCINATED"

    # Extreme contradiction threshold (0.95) -> Contradicted claim falls back to UNVERIFIED
    res_high_contra = verifier.recompute_verification_result(
        sample_claims,
        entailment_thresh=0.55,
        contradiction_thresh=0.95,
        min_similarity_thresh=0.25,
    )
    assert res_high_contra.counts["contradicted"] == 0
    assert res_high_contra.counts["unverified"] == 1
    assert res_high_contra.verdict == "PARTIALLY_GROUNDED"


# =========================================================================
# STAGE 4: BENCHMARK DATASET & EVALUATION INTEGRATION TESTS
# =========================================================================

def test_benchmark_dataset_integrity():
    dataset_path = Path("data/benchmark_dataset.json")
    assert dataset_path.exists(), "Benchmark dataset file must exist"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 30, f"Expected 30 benchmark test cases, got {len(data)}"

    categories = set()
    for item in data:
        assert "id" in item
        assert "category" in item
        assert "domain" in item
        assert "source_text" in item and len(item["source_text"]) > 20
        assert "generated_text" in item and len(item["generated_text"]) > 10
        assert item["ground_truth_label"] in ["GROUNDED", "HALLUCINATED"]
        categories.add(item["category"])

    # Ensure all 4 categories are represented
    assert "Fully Grounded" in categories
    assert "Numerical Distortion" in categories
    assert "Entity/Attribute Swap" in categories
    assert "Extrapolated Fabrication" in categories


def test_evaluator_mini_run(tmp_path):
    # Run evaluator on 2 cases in an isolated tmp directory to test end-to-end integration without touching production metrics
    evaluator = FactLensEvaluator(results_dir=str(tmp_path))
    metrics = evaluator.run_benchmark(limit=2)

    assert "baseline_whole_document" in metrics
    assert "factlens_atomic_claim" in metrics
    assert "category_breakdown" in metrics
    assert metrics["total_samples"] == 2
