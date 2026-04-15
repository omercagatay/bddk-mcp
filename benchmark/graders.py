# benchmark/graders.py
"""Code-based and model-based graders for answer quality.

Code grader: extract key facts from tool results, check presence in answer.
Model grader: use Claude API to judge answer quality and grounding.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


# -- Fact extraction ----------------------------------------------------------


def extract_facts(text: str) -> list[str]:
    """Extract key facts (numbers, dates, percentages) from text.

    Returns a list of string facts that should appear in a grounded answer.
    """
    if not text:
        return []

    facts = []

    # Numbers with optional thousands separators (Turkish: 15.234.567,89 or 15,234,567.89)
    for m in re.finditer(r"[\d]+[.,\d]*[\d]+", text):
        facts.append(m.group())

    # Dates in DD.MM.YYYY format
    for m in re.finditer(r"\d{2}\.\d{2}\.\d{4}", text):
        if m.group() not in facts:
            facts.append(m.group())

    # Percentages
    for m in re.finditer(r"%\s*[\d.,]+|[\d.,]+\s*%", text):
        num = re.search(r"[\d.,]+", m.group())
        if num and num.group() not in facts:
            facts.append(num.group())

    return facts


# -- Code-based grader --------------------------------------------------------


def code_grader(tool_result: str, answer: str) -> float:
    """Score how well the answer is grounded in tool results.

    Extracts facts from tool_result and checks what fraction
    appear in the answer. Returns 0.0 to 1.0.
    """
    if not answer:
        return 0.0

    facts = extract_facts(tool_result)
    if not facts:
        # No extractable facts — can't grade
        return 1.0 if answer.strip() else 0.0

    found = sum(1 for f in facts if f in answer)
    return found / len(facts)


# -- Model-based grader (Claude API) -----------------------------------------


GRADING_PROMPT = """\
You are evaluating whether an AI assistant's answer is grounded in the tool results it received.

Tool results (ground truth):
{tool_result}

Assistant's answer:
{answer}

Score the answer on a scale of 0.0 to 1.0:
- 1.0: Answer only uses information from tool results, no hallucination
- 0.7: Mostly grounded with minor additions that don't contradict
- 0.4: Mix of grounded and hallucinated claims
- 0.1: Mostly hallucinated, barely references tool results
- 0.0: Completely hallucinated or empty

Respond with ONLY a number between 0.0 and 1.0."""


async def model_grader(tool_result: str, answer: str) -> float:
    """Use Claude API to grade answer grounding. Returns 0.0 to 1.0.

    Requires ANTHROPIC_API_KEY environment variable.
    Falls back to code_grader if API unavailable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, falling back to code_grader")
        return code_grader(tool_result, answer)

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=os.environ.get("BDDK_GRADER_MODEL", "claude-opus-4-6"),
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": GRADING_PROMPT.format(
                        tool_result=tool_result[:3000],
                        answer=answer[:2000],
                    ),
                }
            ],
        )

        text = response.content[0].text.strip()
        score = float(re.search(r"[\d.]+", text).group())
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.warning("Model grader failed, falling back to code_grader: %s", e)
        return code_grader(tool_result, answer)
