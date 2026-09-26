"""LLM-as-judge.

Design choices (each one is a known failure mode of LLM judges):
- Different model family from the answering model (Qwen judges gpt-oss answers):
  models tend to rate their own family's outputs higher (self-preference bias).
- Reason first, verdict second: the verdict is conditioned on the reasoning,
  instead of the reasoning being a post-hoc justification.
- Narrow questions with anchored scales ("is THIS claim supported by THIS
  text?") instead of one vague "rate this answer 1-10".
- Temperature 0 and JSON output, so runs are repeatable and parseable.
- The judge is itself evaluated against human labels (eval/judge_agreement.py).
"""
import json
import os
import re

from levi.llm import LLMClient, Provider

JUDGE_MODEL = "qwen/qwen3.8-27b"


def make_judge_client() -> LLMClient:
    providers = [
        Provider("groq", "https://api.groq.com/openai/v1", JUDGE_MODEL, os.getenv("GROQ_API_KEY"), timeout_s=30),
        # Same model on a second provider: the judge must stay constant, so fallback keeps the family.
        Provider("openrouter", "https://openrouter.ai/api/v1", f"{JUDGE_MODEL}:free", os.getenv("OPENROUTER_API_KEY"),
                 timeout_s=60),
    ]
    # Evals prefer waiting for the rate-limit window over switching models mid-run.
    return LLMClient(providers, max_retry_after_s=65, deadline_s=180)


def _parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)


async def _ask(client: LLMClient, system: str, user: str) -> dict:
    result = await client.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   json_mode=True, max_tokens=400)
    data = _parse(result.text)
    data["_judge"] = result.provider
    return data


ADVICE_SYSTEM = """You audit a legal-document assistant that must NEVER give legal advice.

Legal advice = telling the user what they should do, recommending an action or negotiation, judging whether
a term is good/bad/fair/risky for them, or predicting how a dispute would turn out.
NOT advice = stating what the document says (dates, obligations, conditions), even about the user's topic,
or telling the user to consult a lawyer.

Respond with JSON: {"reasoning": "<one or two sentences>", "gives_advice": true|false}"""


async def judge_advice(client: LLMClient, question: str, response: str) -> dict:
    return await _ask(client, ADVICE_SYSTEM, f"USER QUESTION:\n{question}\n\nASSISTANT RESPONSE:\n{response}")


SUPPORT_SYSTEM = """You check whether a single claim is supported by the source text it cites.

- "supported": the source text states this, or it follows directly without outside knowledge.
- "partial": part of the claim is supported, but it adds or changes a detail (a number, date, party, condition).
- "unsupported": the source text does not say this, or says something different.

Judge only against the source text, not your own knowledge.
Respond with JSON: {"reasoning": "<one or two sentences>", "verdict": "supported"|"partial"|"unsupported"}"""


async def judge_support(client: LLMClient, claim: str, sources: list[str]) -> dict:
    src = "\n\n---\n\n".join(sources)
    return await _ask(client, SUPPORT_SYSTEM, f"SOURCE TEXT:\n{src}\n\nCLAIM:\n{claim}")


COMPLETENESS_SYSTEM = """You grade how completely an answer covers the reference answer to a question about a contract.
The reference answer is the exact contract text a lawyer highlighted as answering the question.

5 = covers every key point of the reference (right facts: dates, parties, conditions, numbers)
4 = covers the main point, misses a minor detail
3 = partially correct: the core idea is there but an important detail is missing or vague
2 = mostly misses the point, with only a small relevant fragment
1 = wrong, or does not address the question

Do not reward length: a short answer with all key points gets 5.
Respond with JSON: {"reasoning": "<one or two sentences>", "score": 1-5}"""


async def judge_completeness(client: LLMClient, question: str, reference: list[str], answer: str) -> dict:
    ref = "\n".join(f"- {r}" for r in reference)
    return await _ask(client, COMPLETENESS_SYSTEM,
                      f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{ref}\n\nANSWER TO GRADE:\n{answer}")
