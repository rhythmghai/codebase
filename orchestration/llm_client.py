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
import json
import logging
import concurrent.futures
import tenacity
from dotenv import load_dotenv
load_dotenv()

from common.config import LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# A single shared executor for enforcing a wall-clock timeout around the
# blocking google-genai SDK calls (the SDK itself doesn't expose a timeout
# knob we can rely on across versions) -- without this, a hung upstream
# call blocks the request-handling thread indefinitely, since /query runs
# the pipeline synchronously.
_llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm-call")

# Retries only on the failure classes that are plausibly transient
# (timeout, connection issues) -- a bad API key or malformed request would
# just fail identically three times, so retrying those wastes latency
# without buying anything.
_retry_transient = tenacity.retry(
    reraise=True,
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=0.5, max=4),
    retry=tenacity.retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
)


def _call_with_timeout(fn, *args, **kwargs):
    future = _llm_executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=LLM_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError as e:
        raise TimeoutError(f"LLM call exceeded {LLM_TIMEOUT_SECONDS}s timeout") from e


class BaseLLM:
    def rewrite_query(self, query: str) -> dict:
        raise NotImplementedError

    def generate_answer(self, query: str, context: str, strict: bool = False) -> str:
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

    def generate_answer(self, query: str, context: str, strict: bool = False) -> str:
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

    @_retry_transient
    def _generate_content(self, prompt: str):
        return _call_with_timeout(self.client.models.generate_content, model=self.model_name, contents=prompt)

    def rewrite_query(self, query: str) -> dict:
        prompt = f"""You rewrite a user's question about a codebase into search-friendly variants.
Return ONLY valid JSON, no markdown fences, no preamble, matching this exact schema:
{{"semantic": "<natural language rephrase>", "lexical": "<space-separated likely identifiers/keywords>", "sub_queries": ["<sub-question 1>", "..."]}}

If the question is single-hop, sub_queries should contain just the original question.
If it's multi-hop (e.g. "why does X call Y and what breaks if Y changes"), split it into 2-3 sub-questions.

User question: {query}"""
        resp = self._generate_content(prompt)
        text = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, AttributeError) as e:
            # Small/fast models occasionally return malformed JSON despite
            # the schema instruction -- degrade to the same shape
            # RuleBasedLLM would produce rather than letting an uncaught
            # JSONDecodeError surface as an opaque 500 to the API caller.
            logger.warning("Gemini rewrite_query returned unparseable JSON (%s); using raw query as fallback", e)
            return {"semantic": query, "lexical": query, "sub_queries": [query]}
        if not isinstance(parsed, dict) or "semantic" not in parsed or "lexical" not in parsed:
            logger.warning("Gemini rewrite_query returned JSON missing required fields: %r", parsed)
            return {"semantic": query, "lexical": query, "sub_queries": [query]}
        parsed.setdefault("sub_queries", [query])
        return parsed

    def generate_answer(self, query: str, context: str, strict: bool = False) -> str:
        strict_instruction = (
            "\nYour previous attempt at this answer could not be verified as grounded in the "
            "provided context (it didn't reference any retrieved file path or symbol name). "
            "This time, only state things directly supported by the context below, and "
            "explicitly cite the file path or qualified name backing each claim. If the "
            "context is insufficient to answer, say so explicitly rather than guessing.\n"
            if strict else ""
        )
        prompt = f"""You are a codebase Q&A assistant. Answer the question using ONLY the
provided context (code chunks with file paths and signatures). If the context
doesn't contain enough information, say so explicitly rather than guessing.
Cite file paths and function/class names when relevant.
{strict_instruction}
Context:
{context}

Question: {query}

Answer:"""
        resp = self._generate_content(prompt)
        return resp.text


def get_llm(backend: str | None = None) -> BaseLLM:
    backend = backend or ("gemini" if os.environ.get("GOOGLE_API_KEY") else "rule_based")
    if backend == "gemini":
        return GeminiLLM()
    return RuleBasedLLM()
