"""FactLens: Atomic Claim-Level Hallucination & Grounding Verifier.

A modular 4-stage NLP verification framework that detects hallucinations in AI-generated
text by decomposing text into atomic claims, retrieving dense sentence-level evidence,
and verifying entailment/contradiction via cross-encoder Natural Language Inference.
"""

from src.decomposer import ClaimDecomposer
from src.retriever import EvidenceRetriever
from src.verifier import GroundingVerifier, VerificationResult, ClaimVerification

__version__ = "1.0.0"
__all__ = [
    "ClaimDecomposer",
    "EvidenceRetriever",
    "GroundingVerifier",
    "VerificationResult",
    "ClaimVerification",
]
