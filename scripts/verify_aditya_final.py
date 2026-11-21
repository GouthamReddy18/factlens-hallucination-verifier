import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decomposer import ClaimDecomposer
from src.retriever import EvidenceRetriever
from src.verifier import GroundingVerifier

source = (
    "The Indian Space Research Organisation (ISRO) launched the Aditya-L1 spacecraft aboard "
    "the PSLV-C57 rocket on September 2, 2023. Aditya-L1 represents India's first dedicated "
    "space-based observatory designed to study the Sun from a halo orbit around the Sun-Earth "
    "Lagrangian point 1 (L1), approximately 1.5 million kilometers from Earth. The primary "
    "scientific objectives include observing the solar corona, chromospheric dynamics, and "
    "coronal mass ejections."
)

generated = (
    "ISRO launched the spacecraft on September 2, 2023, using a PSLV-C57 rocket. "
    "Aditya-L1 is India's first dedicated solar observation mission. "
    "The spacecraft will travel approximately 1.5 million kilometers to the Sun-Earth L1 point. "
    "Its primary scientific objective is observing the solar corona and chromospheric dynamics."
)

decomposer = ClaimDecomposer(default_mode="syntactic")
retriever = EvidenceRetriever(device="cpu")
verifier = GroundingVerifier(device="cpu")

claims = decomposer.decompose(generated)
print(f"Decomposed Claims ({len(claims)}):")
for i, c in enumerate(claims, 1):
    print(f"  [{i}] {c}")

retrieval = retriever.retrieve_evidence(source, claims, top_k=2)
result = verifier.verify_pipeline(retrieval, run_baseline=True, source_text=source, generated_text=generated)

print("\n" + "="*70)
print(f"PIPELINE GROUNDING VERIFICATION RESULT")
print(f"Overall Document Verdict: {result.verdict}")
print(f"Grounding Trust Score:   {result.trust_score:.2f}% (Target: > 90.0%)")
print(f"Summary Counts:          {result.counts}")
print("="*70)

for i, c in enumerate(result.claims, 1):
    print(f"\n--- Claim #{i} ---")
    print(f"Claim Text:           \"{c.claim}\"")
    print(f"Single-Sentence Ev:   \"{c.evidence_text}\"")
    print(f"Retrieval Cosine Sim: {c.evidence_similarity:.4f}")
    print(f"Verdict:              {c.verdict}")
    print(f"Softmax Entailment:   {c.probabilities['entailment']*100:.2f}%")
    print(f"Softmax Neutral:      {c.probabilities['neutral']*100:.2f}%")
    print(f"Softmax Contradict:   {c.probabilities['contradiction']*100:.2f}%")
    assert c.verdict == "SUPPORTED", f"Claim {i} was not SUPPORTED: {c.verdict}"
    assert c.probabilities["entailment"] > 0.85, f"Claim {i} entailment too low: {c.probabilities['entailment']}"

assert result.trust_score >= 90.0, f"Trust score below 90%: {result.trust_score}"
print("\n>>> ALL 4 ADITYA-L1 CLAIMS VERIFIED AS 'SUPPORTED' WITH GTS > 90% <<<")
