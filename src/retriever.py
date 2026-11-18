"""Dense evidence retrieval module for FactLens.

Segments reference source text into candidate evidence sentences, computes dense
semantic embeddings using sentence-transformers, and retrieves top-k relevant
sentences for each atomic claim based on cosine similarity.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def ensure_nltk_punkt() -> None:
    """Ensure NLTK sentence tokenizer models are downloaded."""
    import nltk
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK package '{pkg}': {e}")


class EvidenceRetriever:
    """Retrieves the most semantically relevant source sentences for atomic claims.

    Uses a dense bi-encoder ('sentence-transformers/all-MiniLM-L6-v2') to map
    sentences into a shared metric embedding space, computing cosine similarity
    to select the top-k supporting evidence candidates.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        lazy_load: bool = False,
    ) -> None:
        """Initialize the EvidenceRetriever.

        Args:
            model_name: Hugging Face model identifier for sentence encoding.
            device: 'cpu', 'cuda', or None (auto-detected).
            lazy_load: If True, defer model loading until first retrieval call.
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        ensure_nltk_punkt()

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load sentence-transformers bi-encoder model."""
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading SentenceTransformer '{self.model_name}' on {self.device}...")
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def model(self):
        """Property that ensures model is loaded."""
        if self._model is None:
            self._load_model()
        return self._model

    def split_sentences(self, text: str) -> List[str]:
        """Split a source text document into clean, individual sentences."""
        import nltk

        if not text or not text.strip():
            return []

        raw_sentences = nltk.sent_tokenize(text.strip())
        sentences: List[str] = []
        for s in raw_sentences:
            s_clean = s.strip()
            if s_clean and len(s_clean.split()) >= 2:
                sentences.append(s_clean)

        return sentences if sentences else [text.strip()]

    def retrieve_evidence(
        self,
        source_text: str,
        claims: List[str],
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k highest-scoring source sentences for each claim.

        Args:
            source_text: Full reference/source document string.
            claims: List of atomic claims to ground.
            top_k: Number of highest-similarity sentences to return per claim.

        Returns:
            List of dicts, each containing:
                - 'claim': Original claim text.
                - 'evidence_sentences': List of top-k retrieved source sentences.
                - 'evidence_scores': List of cosine similarity scores (floats).
                - 'combined_evidence': Combined text of the top evidence sentences.
                - 'top_score': Highest similarity score among retrieved sentences.
        """
        if not claims:
            return []

        source_sentences = self.split_sentences(source_text)
        if not source_sentences:
            return [
                {
                    "claim": c,
                    "evidence_sentences": [],
                    "evidence_scores": [],
                    "combined_evidence": "",
                    "top_score": 0.0,
                }
                for c in claims
            ]

        # Compute embeddings
        source_embeddings = self.model.encode(
            source_sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        claim_embeddings = self.model.encode(
            claims,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        results: List[Dict[str, Any]] = []

        # For normalized embeddings, cosine similarity is simply the inner product
        sim_matrix = np.dot(claim_embeddings, source_embeddings.T)  # shape: (n_claims, n_sentences)

        k = min(top_k, len(source_sentences))

        for idx, claim in enumerate(claims):
            scores = sim_matrix[idx]
            # Get indices of top-k scores in descending order
            top_indices = np.argsort(scores)[::-1][:k]

            matched_sentences = [source_sentences[i] for i in top_indices]
            matched_scores = [float(scores[i]) for i in top_indices]
            top_score = matched_scores[0] if matched_scores else 0.0

            results.append({
                "claim": claim,
                "evidence_sentences": matched_sentences,
                "evidence_scores": matched_scores,
                "combined_evidence": " ".join(matched_sentences),
                "top_score": round(top_score, 4),
            })

        return results
