"""
Grounded answer generation.

The generator only ever sees passages that passed the evidence gate. Two modes:

* LLM (Gemini or any OpenAI-compatible endpoint), used when an API key is configured.
  The prompt restricts the model to the supplied passages and requires a [n] citation
  after every claim.
* Extractive, the default. It quotes the sentences of each passage that best match the
  question. No model is involved, so the answer cannot contain anything that is not
  in the label text, and the whole pipeline runs offline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import httpx

from core.chunking import ClinicalChunk, split_sentences
from core.sparse import tokenize

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about drug interactions using FDA prescribing information.

Rules:
- Use only the numbered passages provided. Do not add facts from memory.
- Put a citation such as [1] or [2] after every claim, referring to the passage number.
- If the passages do not answer the question, reply exactly: INSUFFICIENT_EVIDENCE
- Be concise: state the interaction, its mechanism if the passages give one, and the
  recommended action if the passages give one."""


@dataclass
class GeneratedAnswer:
    text: str
    mode: str  # "gemini" | "openai" | "extractive"


def build_prompt(
    query: str, passages: Sequence[Tuple[ClinicalChunk, float]], graph_triples: Sequence[str]
) -> str:
    lines = [f"Question: {query}", "", "Passages:"]
    for i, (chunk, _) in enumerate(passages, start=1):
        lines.append(f"[{i}] {chunk.title} | {chunk.section}\n{chunk.content}\n")
    if graph_triples:
        lines.append("Relations extracted from these labels (context only, cite the passages):")
        lines += [f"- {t}" for t in graph_triples[:12]]
    return "\n".join(lines)


def extractive_answer(
    query: str, passages: Sequence[Tuple[ClinicalChunk, float]], sentences_per_passage: int = 2
) -> str:
    """Quotes the sentences in each passage that share the most terms with the query."""
    query_terms = set(tokenize(query))
    blocks: List[str] = []
    for i, (chunk, _) in enumerate(passages, start=1):
        sentences = split_sentences(chunk.content)
        scored = sorted(
            range(len(sentences)),
            key=lambda idx: len(query_terms & set(tokenize(sentences[idx]))),
            reverse=True,
        )[:sentences_per_passage]
        quote = " ".join(sentences[idx] for idx in sorted(scored))
        blocks.append(f"{quote} [{i}]")
    return "\n\n".join(blocks)


class AnswerGenerator:
    def __init__(
        self,
        gemini_model: str = "gemini-2.5-flash",
        openai_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 700,
    ):
        self.gemini_model = os.getenv("GEMINI_MODEL", gemini_model)
        self.openai_model = os.getenv("OPENAI_MODEL", openai_model)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def mode(self) -> str:
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "extractive"

    def _call_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        res = httpx.post(
            url,
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                },
            },
            timeout=30.0,
        )
        res.raise_for_status()
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt: str) -> str:
        res = httpx.post(
            f"{self.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            json={
                "model": self.openai_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=30.0,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def generate(
        self,
        query: str,
        passages: Sequence[Tuple[ClinicalChunk, float]],
        graph_triples: Sequence[str] = (),
    ) -> GeneratedAnswer:
        mode = self.mode
        if mode != "extractive":
            prompt = build_prompt(query, passages, graph_triples)
            try:
                text: str | None = (
                    self._call_gemini(prompt) if mode == "gemini" else self._call_openai(prompt)
                )
                if text and text.strip():
                    return GeneratedAnswer(text.strip(), mode)
            except (httpx.HTTPError, KeyError, IndexError) as exc:
                logger.warning("%s generation failed (%s); using extractive answer", mode, exc)
        return GeneratedAnswer(extractive_answer(query, passages), "extractive")
