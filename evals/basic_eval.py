"""
Evals 101: The Simplest Possible Eval — Exact Match Grading

An "eval" answers one question: how do we know if our LLM app actually
works? Without one, every change is a guess — "looks fine to me" is not
a measurement.

The basic recipe for any eval:
  1. TEST CASES  — a fixed list of (input, expected_output) pairs
  2. RUN         — send each input to the model, capture its output
  3. GRADE       — compare actual vs expected with a scoring function
  4. REPORT      — pass rate + which cases failed, so you can act on it

This script uses the simplest possible grader: exact/contains string
match. It works great for facts with one right answer (math, lookups,
short factual recall) and falls apart for open-ended answers — that's
the motivation for the LLM-as-judge eval that comes next in this module.

Run:
    python basic_eval.py
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# TEST CASES — input + the expected answer
# ─────────────────────────────────────────────

TEST_CASES = [
    {"input": "What is 12 * 7? Reply with just the number.", "expected": "84"},
    {"input": "What is the capital of France? Reply with just the city name.", "expected": "Paris"},
    {"input": "How many continents are there on Earth? Reply with just the number.", "expected": "7"},
    {"input": "What is the chemical symbol for gold? Reply with just the symbol.", "expected": "Au"},
    {"input": "What is 100 divided by 4? Reply with just the number.", "expected": "25"},
]


# ─────────────────────────────────────────────
# RUN — call the model
# ─────────────────────────────────────────────

def call_model(prompt):
    """Send one prompt to the model and return its text response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# GRADE — exact match grader
# ─────────────────────────────────────────────

def grade(actual, expected):
    """Pass if the expected answer appears in the model's output (case-insensitive)."""
    return expected.lower() in actual.lower()


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def run_eval(test_cases):
    """Run every test case, grade it, and return the results."""
    results = []
    for case in test_cases:
        actual = call_model(case["input"])
        passed = grade(actual, case["expected"])
        results.append({**case, "actual": actual, "passed": passed})
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("   BASIC EVAL — Exact Match Grading")
    print("=" * 60)

    results = run_eval(TEST_CASES)

    print()
    for i, r in enumerate(results, 1):
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] Test {i}: {r['input']}")
        print(f"         expected: {r['expected']!r}  |  got: {r['actual']!r}")

    passed_count = sum(r["passed"] for r in results)
    total = len(results)
    score = passed_count / total * 100

    print(f"\n  Score: {passed_count}/{total} ({score:.0f}%)\n")

    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  This grader only works because every test case has a   │
  │  single correct string. Ask "explain photosynthesis"    │
  │  instead and exact match breaks — there's no one right  │
  │  string to compare against.                             │
  │                                                         │
  │  NEXT STEP: use a second LLM call as the grader          │
  │  ("LLM-as-judge") for open-ended, free-form answers.     │
  └─────────────────────────────────────────────────────────┘
""")
