"""Cross-Encoder NLI Verification Engine for FactLens.

Evaluates premise-hypothesis pairs using MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli,
mapping entailment, neutral, and contradiction probabilities to structured grounding
verdicts (SUPPORTED, CONTRADICTED, UNVERIFIED) and calculating a holistic Grounding Trust Score.

Includes dynamic PyTorch INT8 quantization on CPU, thread optimization,
batched tensor inference, and decoupled zero-lag verdict recomputation.
"""

from __future__ import annotations

import os
import gc
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaimVerification:
    """Individual atomic claim verification record."""
    claim: str
    evidence_text: str
    evidence_similarity: float
    verdict: str  # 'SUPPORTED', 'CONTRADICTED', or 'UNVERIFIED'
    probabilities: Dict[str, float]  # 'entailment', 'neutral', 'contradiction'
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Aggregated document-level verification output."""
    claims: List[ClaimVerification]
    trust_score: float  # 0.0 to 100.0
    verdict: str  # 'FULLY_GROUNDED', 'PARTIALLY_GROUNDED', or 'HALLUCINATED'
    counts: Dict[str, int]  # 'supported', 'contradicted', 'unverified', 'total'
    latency_ms: float
    baseline_whole_doc: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["claims"] = [c.to_dict() if isinstance(c, ClaimVerification) else c for c in self.claims]
        return res


class GroundingVerifier:
    """Cross-encoder Natural Language Inference verifier for atomic claims."""

    DEFAULT_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        entailment_threshold: float = 0.55,
        contradiction_threshold: float = 0.45,
        min_similarity_threshold: float = 0.25,
        lazy_load: bool = False,
        quantize: bool = True,
    ) -> None:
        """Initialize GroundingVerifier.

        Args:
            model_name: Hugging Face model checkpoint for NLI cross-encoder.
            device: 'cpu', 'cuda', or None (auto-detect).
            entailment_threshold: Minimum entailment probability required for SUPPORTED.
            contradiction_threshold: Minimum contradiction probability required for CONTRADICTED.
            min_similarity_threshold: Evidence cosine similarity below which claim is UNVERIFIED.
            lazy_load: If True, defer model loading until first inference call.
            quantize: If True and on CPU, apply PyTorch dynamic INT8 quantization.
        """
        self.model_name = model_name
        self.device = device
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.quantize = quantize
        self.is_quantized: bool = False
        self._tokenizer = None
        self._model = None
        self._base_model = None

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and sequence classification model with optional INT8 quantization."""
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU Thread configuration for faster linear/attention operations
        if self.device == "cpu":
            num_threads = min(4, os.cpu_count() or 4)
            try:
                torch.set_num_threads(num_threads)
            except Exception:
                pass

        logger.info(f"Loading NLI model '{self.model_name}' on {self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        base_model = base_model.float()
        base_model.to(self.device)
        base_model.eval()
        self._base_model = base_model

        # Dynamic INT8 Quantization on CPU
        if self.device == "cpu" and self.quantize:
            try:
                gc.collect()
                logger.info("Applying dynamic INT8 quantization to DeBERTa-v3 Linear layers...")
                self._model = torch.quantization.quantize_dynamic(
                    base_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                self.is_quantized = True
                logger.info("DeBERTa-v3 successfully dynamically quantized to INT8.")
            except Exception as e:
                logger.warning(f"INT8 dynamic quantization skipped, using FP32 ({e}).")
                self._model = self._base_model
                self.is_quantized = False
        else:
            self._model = base_model
            self.is_quantized = False

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _run_forward(self, inputs: Dict[str, Any]) -> Any:
        """Execute forward pass with automatic fallback to unquantized FP32 model if quantized execution fails."""
        import torch

        context_manager = getattr(torch, "inference_mode", torch.no_grad)
        try:
            with context_manager():
                return self.model(**inputs)
        except RuntimeError as e:
            err_msg = str(e)
            if self.is_quantized or ("qlinear_dynamic" in err_msg or "data type" in err_msg or "ONEDNN" in err_msg):
                logger.warning(
                    f"Dynamic quantization forward pass failed ({err_msg}). "
                    "Automatically falling back to unquantized float32 DeBERTa-v3 CPU execution."
                )
                if self._base_model is not None:
                    self._model = self._base_model
                else:
                    from transformers import AutoModelForSequenceClassification
                    self._model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                    ).float().to(self.device).eval()
                    self._base_model = self._model
                self.is_quantized = False
                with context_manager():
                    return self._model(**inputs)
            raise e

    def _get_label_mapping(self) -> Dict[str, int]:
        """Dynamically extract label mapping from model config."""
        id2label = getattr(self.model.config, "id2label", {0: "entailment", 1: "neutral", 2: "contradiction"})
        mapping = {}
        for idx, name in id2label.items():
            name_lower = str(name).lower()
            if "entail" in name_lower:
                mapping["entailment"] = int(idx)
            elif "contra" in name_lower:
                mapping["contradiction"] = int(idx)
            elif "neut" in name_lower:
                mapping["neutral"] = int(idx)
        return mapping

    def evaluate_verdict(
        self,
        probabilities: Dict[str, float],
        evidence_similarity: float,
        entailment_thresh: Optional[float] = None,
        contradiction_thresh: Optional[float] = None,
        min_similarity_thresh: Optional[float] = None,
    ) -> tuple[str, str]:
        """Map probabilities and similarity to discrete verdict and rationale without re-running model."""
        e_th = entailment_thresh if entailment_thresh is not None else self.entailment_threshold
        c_th = contradiction_thresh if contradiction_thresh is not None else self.contradiction_threshold
        s_th = min_similarity_thresh if min_similarity_thresh is not None else self.min_similarity_threshold

        p_entail = probabilities.get("entailment", 0.0)
        p_neutral = probabilities.get("neutral", 0.0)
        p_contra = probabilities.get("contradiction", 0.0)

        if evidence_similarity < s_th:
            return (
                "UNVERIFIED",
                f"Retrieved evidence similarity ({evidence_similarity:.3f}) is below threshold ({s_th:.2f}); claim cannot be verified against source.",
            )

        if p_entail >= e_th and p_entail > p_contra:
            return "SUPPORTED", f"Supported by source evidence with {p_entail:.1%} entailment confidence."
        elif p_contra >= c_th and p_contra > p_entail:
            return "CONTRADICTED", f"Contradicts source evidence with {p_contra:.1%} contradiction confidence."
        else:
            return "UNVERIFIED", f"Insufficient conclusive evidence in source (neutral={p_neutral:.1%}, entail={p_entail:.1%}, contra={p_contra:.1%})."

    def verify_pair(
        self,
        premise: str,
        hypothesis: str,
        evidence_similarity: float = 1.0,
    ) -> ClaimVerification:
        """Verify a single (premise=evidence, hypothesis=claim) pair."""
        import torch

        if not premise.strip() or evidence_similarity < self.min_similarity_threshold:
            return ClaimVerification(
                claim=hypothesis,
                evidence_text=premise,
                evidence_similarity=evidence_similarity,
                verdict="UNVERIFIED",
                probabilities={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
                explanation="Retrieved evidence similarity is below threshold or absent; claim cannot be verified against source.",
            )

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._run_forward(inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

        label_map = self._get_label_mapping()
        p_entail = float(probs[label_map.get("entailment", 0)].item())
        p_neutral = float(probs[label_map.get("neutral", 1)].item())
        p_contra = float(probs[label_map.get("contradiction", 2)].item())

        prob_dict = {
            "entailment": round(p_entail, 4),
            "neutral": round(p_neutral, 4),
            "contradiction": round(p_contra, 4),
        }

        verdict, explanation = self.evaluate_verdict(prob_dict, evidence_similarity)

        return ClaimVerification(
            claim=hypothesis,
            evidence_text=premise,
            evidence_similarity=evidence_similarity,
            verdict=verdict,
            probabilities=prob_dict,
            explanation=explanation,
        )

    def verify_batch(
        self,
        pairs: List[Dict[str, Any]],
    ) -> List[ClaimVerification]:
        """Verify multiple (premise, hypothesis, similarity) pairs in a single batched inference call.

        Optimized with max_length=256 and torch.inference_mode.
        """
        import torch

        if not pairs:
            return []

        verifications: List[Optional[ClaimVerification]] = [None] * len(pairs)
        eligible_indices: List[int] = []
        eligible_premises: List[str] = []
        eligible_hypotheses: List[str] = []

        for idx, item in enumerate(pairs):
            premise = item.get("premise", "").strip()
            hypothesis = item.get("hypothesis", "").strip()
            similarity = float(item.get("evidence_similarity", 1.0))

            if not premise or similarity < self.min_similarity_threshold:
                verifications[idx] = ClaimVerification(
                    claim=hypothesis,
                    evidence_text=premise,
                    evidence_similarity=similarity,
                    verdict="UNVERIFIED",
                    probabilities={"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0},
                    explanation="Retrieved evidence similarity is below threshold or absent; claim cannot be verified against source.",
                )
            else:
                eligible_indices.append(idx)
                eligible_premises.append(premise)
                eligible_hypotheses.append(hypothesis)

        if eligible_indices:
            inputs = self.tokenizer(
                eligible_premises,
                eligible_hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            outputs = self._run_forward(inputs)
            all_probs = torch.softmax(outputs.logits, dim=-1)

            label_map = self._get_label_mapping()

            for i, original_idx in enumerate(eligible_indices):
                probs = all_probs[i]
                p_entail = float(probs[label_map.get("entailment", 0)].item())
                p_neutral = float(probs[label_map.get("neutral", 1)].item())
                p_contra = float(probs[label_map.get("contradiction", 2)].item())

                prob_dict = {
                    "entailment": round(p_entail, 4),
                    "neutral": round(p_neutral, 4),
                    "contradiction": round(p_contra, 4),
                }

                sim_val = float(pairs[original_idx].get("evidence_similarity", 1.0))
                verdict, explanation = self.evaluate_verdict(prob_dict, sim_val)

                verifications[original_idx] = ClaimVerification(
                    claim=eligible_hypotheses[i],
                    evidence_text=eligible_premises[i],
                    evidence_similarity=sim_val,
                    verdict=verdict,
                    probabilities=prob_dict,
                    explanation=explanation,
                )

        return [v for v in verifications if v is not None]

    def verify_pipeline(
        self,
        retrieval_results: List[Dict[str, Any]],
        run_baseline: bool = False,
        source_text: str = "",
        generated_text: str = "",
    ) -> VerificationResult:
        """Run verification over retrieved claim-evidence candidate pairs using batched inference."""
        start_time = time.perf_counter()

        pairs_to_verify = [
            {
                "premise": item.get("combined_evidence", ""),
                "hypothesis": item.get("claim", ""),
                "evidence_similarity": item.get("top_score", 0.0),
            }
            for item in retrieval_results
        ]

        claims_verified = self.verify_batch(pairs_to_verify)

        # Compute summary counts
        counts = {
            "supported": sum(1 for c in claims_verified if c.verdict == "SUPPORTED"),
            "contradicted": sum(1 for c in claims_verified if c.verdict == "CONTRADICTED"),
            "unverified": sum(1 for c in claims_verified if c.verdict == "UNVERIFIED"),
            "total": len(claims_verified),
        }

        # Calculate Grounding Trust Score (0 - 100%)
        trust_score = self._compute_trust_score(claims_verified)

        # High-level verdict
        if counts["contradicted"] > 0:
            doc_verdict = "HALLUCINATED"
        elif counts["unverified"] > 0 and counts["supported"] == 0:
            doc_verdict = "UNVERIFIED"
        elif counts["unverified"] > 0:
            doc_verdict = "PARTIALLY_GROUNDED"
        else:
            doc_verdict = "FULLY_GROUNDED"

        baseline_result = None
        if run_baseline and source_text and generated_text:
            baseline_result = self.verify_whole_document(source_text, generated_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return VerificationResult(
            claims=claims_verified,
            trust_score=trust_score,
            verdict=doc_verdict,
            counts=counts,
            latency_ms=round(elapsed_ms, 2),
            baseline_whole_doc=baseline_result,
        )

    def recompute_verification_result(
        self,
        raw_claims: List[Dict[str, Any]],
        entailment_thresh: float,
        contradiction_thresh: float,
        min_similarity_thresh: float,
        baseline_info: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
    ) -> VerificationResult:
        """Zero-lag recomputation of verdicts and Grounding Trust Score from stored raw probabilities."""
        updated_claims: List[ClaimVerification] = []

        for item in raw_claims:
            claim_text = item.get("claim", "")
            ev_text = item.get("evidence_text", "")
            ev_sim = float(item.get("evidence_similarity", 0.0))
            probs = item.get("probabilities", {"entailment": 0.0, "neutral": 1.0, "contradiction": 0.0})

            v, expl = self.evaluate_verdict(
                probabilities=probs,
                evidence_similarity=ev_sim,
                entailment_thresh=entailment_thresh,
                contradiction_thresh=contradiction_thresh,
                min_similarity_thresh=min_similarity_thresh,
            )

            updated_claims.append(
                ClaimVerification(
                    claim=claim_text,
                    evidence_text=ev_text,
                    evidence_similarity=ev_sim,
                    verdict=v,
                    probabilities=probs,
                    explanation=expl,
                )
            )

        counts = {
            "supported": sum(1 for c in updated_claims if c.verdict == "SUPPORTED"),
            "contradicted": sum(1 for c in updated_claims if c.verdict == "CONTRADICTED"),
            "unverified": sum(1 for c in updated_claims if c.verdict == "UNVERIFIED"),
            "total": len(updated_claims),
        }

        trust_score = self._compute_trust_score(updated_claims)

        if counts["contradicted"] > 0:
            doc_verdict = "HALLUCINATED"
        elif counts["unverified"] > 0 and counts["supported"] == 0:
            doc_verdict = "UNVERIFIED"
        elif counts["unverified"] > 0:
            doc_verdict = "PARTIALLY_GROUNDED"
        else:
            doc_verdict = "FULLY_GROUNDED"

        return VerificationResult(
            claims=updated_claims,
            trust_score=trust_score,
            verdict=doc_verdict,
            counts=counts,
            latency_ms=round(latency_ms, 2),
            baseline_whole_doc=baseline_info,
        )

    def _compute_trust_score(self, claims: List[ClaimVerification]) -> float:
        """Calculate weighted Grounding Trust Score (0.0 to 100.0%).

        Penalizes contradiction heavily (-1.5x) while rewarding supported claims
        weighted by their entailment probability.
        """
        if not claims:
            return 0.0

        raw_score = 0.0
        for c in claims:
            if c.verdict == "SUPPORTED":
                raw_score += 1.0 * c.probabilities.get("entailment", 0.0)
            elif c.verdict == "UNVERIFIED":
                raw_score += 0.20 * (1.0 - c.probabilities.get("contradiction", 0.0))
            elif c.verdict == "CONTRADICTED":
                raw_score -= 1.50 * c.probabilities.get("contradiction", 0.0)

        normalized = (raw_score / len(claims)) * 100.0
        return max(0.0, min(100.0, round(normalized, 1)))

    def verify_whole_document(
        self,
        source_text: str,
        generated_text: str,
    ) -> Dict[str, Any]:
        """Naive whole-text baseline NLI: passes entire premise and hypothesis."""
        import torch

        inputs = self.tokenizer(
            source_text,
            generated_text,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self._run_forward(inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        label_map = self._get_label_mapping()
        p_entail = float(probs[label_map.get("entailment", 0)].item())
        p_neutral = float(probs[label_map.get("neutral", 1)].item())
        p_contra = float(probs[label_map.get("contradiction", 2)].item())

        if p_entail >= self.entailment_threshold and p_entail > p_contra:
            verdict = "SUPPORTED"
        elif p_contra >= self.contradiction_threshold and p_contra > p_entail:
            verdict = "CONTRADICTED"
        else:
            verdict = "UNVERIFIED"

        return {
            "verdict": verdict,
            "probabilities": {
                "entailment": round(p_entail, 4),
                "neutral": round(p_neutral, 4),
                "contradiction": round(p_contra, 4),
            },
            "is_hallucinated": verdict != "SUPPORTED",
        }
