"""
LLM client used for exactly two things in this pipeline: query rewriting
and final answer generation. Both are single, scoped, fixed-prompt calls --
never an open-ended agent deciding what to do next.

RuleBasedLLM is a deterministic fallback so the pipeline is fully testable
without an API key (useful for this sandbox and for CI). GeminiLLM is the
real backend -- same model family already used in CareerRadar
(gemini-2.5-flash-lite), so no new API surface to learn.
"""

import os
import re
from dotenv import load_dotenv
load_dotenv()


class BaseLLM:
    def rewrite_query(self, query: str) -> dict:
        raise NotImplementedError

    def generate_answer(self, query: str, context: str) -> str:
        raise NotImplementedError


class RuleBasedLLM(BaseLLM):
    """
    Deterministic stand-in used when no GOOGLE_API_KEY is set. Not a
    real substitute for an LLM's rewriting ability, but it keeps the
    pipeline runnable end-to-end and makes the seam where the real
    LLM call belongs obvious.
    """

    _CODE_TERM_PATTERN = re.compile(r"\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b|\b[a-z_]+_[a-z_]+\b")

    def rewrite_query(self, query: str) -> dict:
        # "semantic" variant: query as-is, for the embedding search
        # "lexical" variant: pull out anything that looks like an identifier
        # (CamelCase or snake_case) since that's what BM25 should key on
        identifiers = self._CODE_TERM_PATTERN.findall(query)
        lexical = " ".join(identifiers) if identifiers else query
        return {"semantic": query, "lexical": lexical, "sub_queries": [query]}

    def generate_answer(self, query: str, context: str) -> str:
        return (
            "[RuleBasedLLM stand-in -- set GOOGLE_API_KEY to use Gemini for real generation]\n\n"
            f"Retrieved context for: {query}\n\n{context[:800]}"
        )


class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash-lite"):
        from google import genai
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def rewrite_query(self, query: str) -> dict:
        prompt = f"""You rewrite a user's question about a codebase into search-friendly variants.
Return ONLY valid JSON, no markdown fences, no preamble, matching this exact schema:
{{"semantic": "<natural language rephrase>", "lexical": "<space-separated likely identifiers/keywords>", "sub_queries": ["<sub-question 1>", "..."]}}

If the question is single-hop, sub_queries should contain just the original question.
If it's multi-hop (e.g. "why does X call Y and what breaks if Y changes"), split it into 2-3 sub-questions.

User question: {query}"""
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        import json
        text = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(text)

    def generate_answer(self, query: str, context: str) -> str:
        prompt = f"""You are a codebase Q&A assistant. Answer the question using ONLY the
provided context (code chunks with file paths and signatures). If the context
doesn't contain enough information, say so explicitly rather than guessing.
Cite file paths and function/class names when relevant.

Context:
{context}

Question: {query}

Answer:"""
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return resp.text


def get_llm(backend: str | None = None) -> BaseLLM:
    backend = backend or ("gemini" if os.environ.get("GOOGLE_API_KEY") else "rule_based")
    if backend == "gemini":
        return GeminiLLM()
    return RuleBasedLLM()
