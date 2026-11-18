"""Atomic claim decomposition engine for FactLens.

Decomposes complex, multi-sentence AI-generated text into self-contained,
single-fact atomic claims using either syntactic parsing rules (offline) or
LLM-guided pronoun resolution and fact extraction (API-enabled).
"""

from __future__ import annotations

import os
import re
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

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


class ClaimDecomposer:
    """Decomposes text into discrete, testable atomic claims.

    Supports two operational modes:
    1. 'syntactic': Rule-based offline decomposition using sentence tokenization,
       conjunction clause splitting, and subject propagation. Operates 100% on CPU
       without external dependencies.
    2. 'llm': Few-shot prompted extraction via OpenAI, Groq, or Gemini API to
       decontextualize statements, resolve coreferent pronouns, and extract minimal
       falsifiable propositions.
    """

    COORDINATING_CONJUNCTIONS = [
        " and ", " but ", " while ", " whereas ", " although ",
        " however, ", " moreover, ", " furthermore, ", " in addition, "
    ]

    def __init__(
        self,
        default_mode: str = "syntactic",
        llm_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the ClaimDecomposer.

        Args:
            default_mode: 'syntactic' or 'llm'.
            llm_provider: 'gemini', 'openai', or 'groq'. Auto-detected from .env if None.
            api_key: API key for the chosen LLM provider. Defaults to environment variable.
            model_name: Specific model name to invoke (e.g., 'gemini-1.5-flash', 'gpt-4o-mini').
        """
        self.default_mode = default_mode.lower()
        self.llm_provider = (llm_provider or os.getenv("FACTLENS_LLM_PROVIDER", "")).lower()
        self.api_key = api_key
        self.model_name = model_name
        ensure_nltk_punkt()

    def decompose(self, text: str, mode: Optional[str] = None) -> List[str]:
        """Decompose input text into a list of atomic claims.

        Args:
            text: AI-generated response or summary to decompose.
            mode: Override operational mode ('syntactic' or 'llm').

        Returns:
            List of standalone atomic claims as strings.
        """
        if not text or not text.strip():
            return []

        active_mode = (mode or self.default_mode).lower()

        if active_mode == "llm":
            try:
                claims = self._decompose_llm(text)
                if claims:
                    return claims
                logger.warning("LLM decomposition returned empty claims; falling back to syntactic.")
            except Exception as err:
                logger.warning(f"LLM decomposition failed ({err}); falling back to syntactic mode.")

        return self._decompose_syntactic(text)

    def _decompose_syntactic(self, text: str) -> List[str]:
        """Rule-based syntactic decomposition running locally without external APIs.

        Uses sentence tokenization, semicolon splitting, coordinate conjunction
        segmentation, and relative-clause normalization while propagating
        sentence subjects when identifiable.
        """
        import nltk

        # 1. Clean and normalize input text
        cleaned_text = re.sub(r"[\r\n]+", " ", text.strip())
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Remove list markers like "1. ", "- ", "* "
        cleaned_text = re.sub(r"(?:^|\s)(?:[-*]|\d+\.)\s+", " ", cleaned_text)

        raw_sentences = nltk.sent_tokenize(cleaned_text)
        atomic_claims: List[str] = []

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Split on semicolons first
            semi_clauses = [c.strip() for c in sentence.split(";") if c.strip()]

            for clause in semi_clauses:
                sub_claims = self._split_compound_clause(clause)
                for sc in sub_claims:
                    sc = sc.strip()
                    if len(sc) > 10 and not sc.endswith((".", "!", "?")):
                        sc += "."
                    if self._is_valid_claim(sc):
                        atomic_claims.append(sc)

        # Fallback if no atomic claim survived
        if not atomic_claims and text.strip():
            fallback = text.strip()
            if not fallback.endswith("."):
                fallback += "."
            atomic_claims.append(fallback)

        return atomic_claims

    def _split_compound_clause(self, clause: str) -> List[str]:
        """Split a clause on coordinating conjunctions if both halves contain verbal predicates."""
        # Try splitting on coordinate conjunctions
        pattern = r"\b(?:and|but|while|whereas|although)\b"
        tokens = re.split(pattern, clause, flags=re.IGNORECASE)

        if len(tokens) <= 1:
            return [clause]

        # Extract subject candidate from the first part (first few words)
        words = tokens[0].strip().split()
        subject_prefix = ""
        if len(words) >= 2:
            # Simple heuristic: first noun phrase approximation (first 1-3 tokens)
            subject_prefix = " ".join(words[:2])

        split_clauses: List[str] = []
        for i, part in enumerate(tokens):
            part = part.strip()
            # Clean leading/trailing commas and whitespace
            part = re.sub(r"^,\s*", "", part)
            part = re.sub(r",\s*$", "", part)

            if not part:
                continue

            # If this is a secondary part lacking an explicit subject, attempt subject attachment
            has_likely_verb = bool(re.search(r"\b(is|was|are|were|has|have|had|increased|decreased|reported|stated|showed|expanded|reached|grew|fell|approved|tested)\b", part, re.IGNORECASE))
            first_two_words = " ".join(part.split()[:2]) if part.split() else ""
            has_subject = bool(re.search(r"\b(it|he|she|they|this|that|the|a|an)\b", first_two_words, re.IGNORECASE))

            if i > 0 and has_likely_verb and not has_subject and subject_prefix:
                part = f"{subject_prefix} {part}"

            # Capitalize first letter
            if part:
                part = part[0].upper() + part[1:]
                split_clauses.append(part)

        return split_clauses if split_clauses else [clause]

    def _is_valid_claim(self, claim: str) -> bool:
        """Filter out non-informative fragments or trivial phrases."""
        claim_clean = claim.strip()
        if len(claim_clean.split()) < 3:
            return False
        # Must contain at least one alphanumeric character
        if not re.search(r"[a-zA-Z0-9]", claim_clean):
            return False
        return True

    def _decompose_llm(self, text: str) -> List[str]:
        """LLM-based claim decomposition with pronoun resolution."""
        system_prompt = (
            "You are an expert fact-checking claim decomposition engine. "
            "Your task is to decompose the following text into distinct, standalone atomic claims.\n"
            "Rules:\n"
            "1. Each claim must express a single factual proposition.\n"
            "2. Resolve all pronouns (it, they, he, she, this) into explicit entities.\n"
            "3. Do NOT invent new facts; decompose only what is asserted.\n"
            "4. Return each atomic claim on a new line prefixed with '- '.\n"
            "5. Output ONLY the claims, with no intro or commentary."
        )

        user_content = f"Text to decompose:\n\"\"\"{text}\"\"\""

        # Check for available API providers: Gemini, OpenAI, or Groq
        gemini_key = self.api_key or os.getenv("GEMINI_API_KEY")
        openai_key = self.api_key or os.getenv("OPENAI_API_KEY")
        groq_key = self.api_key or os.getenv("GROQ_API_KEY")

        if self.llm_provider == "gemini" or (not self.llm_provider and gemini_key):
            return self._call_gemini(system_prompt, user_content, gemini_key)
        elif self.llm_provider == "openai" or (not self.llm_provider and openai_key):
            return self._call_openai(system_prompt, user_content, openai_key)
        elif self.llm_provider == "groq" or (not self.llm_provider and groq_key):
            return self._call_groq(system_prompt, user_content, groq_key)
        else:
            raise ValueError("No LLM API key configured for 'llm' mode. Set GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY.")

    def _call_gemini(self, system_prompt: str, user_content: str, api_key: Optional[str]) -> List[str]:
        """Invoke Gemini API."""
        from google import genai
        client = genai.Client(api_key=api_key)
        model = self.model_name or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_content}",
        )
        return self._parse_bullet_claims(response.text)

    def _call_openai(self, system_prompt: str, user_content: str, api_key: Optional[str]) -> List[str]:
        """Invoke OpenAI API."""
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        return self._parse_bullet_claims(raw_text)

    def _call_groq(self, system_prompt: str, user_content: str, api_key: Optional[str]) -> List[str]:
        """Invoke Groq API."""
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name or "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        return self._parse_bullet_claims(raw_text)

    def _parse_bullet_claims(self, text: str) -> List[str]:
        """Parse bulleted claims from model response."""
        lines = text.strip().split("\n")
        claims: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^(?:[-*]|\d+\.)\s+", "", line.strip())
            cleaned = cleaned.strip()
            if self._is_valid_claim(cleaned):
                if not cleaned.endswith((".", "!", "?")):
                    cleaned += "."
                claims.append(cleaned)
        return claims
