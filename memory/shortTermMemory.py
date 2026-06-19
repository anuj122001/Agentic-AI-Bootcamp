"""
Script 2: Short-Term Memory Strategies — Managing the Context Window

The naive approach (send everything) breaks at scale. This script implements
three strategies to keep the conversation going without blowing up tokens:

  1. SLIDING WINDOW    — keep only the last K messages (simple, lossy)
  2. SUMMARIZATION     — LLM summarizes old turns into a paragraph
  3. HYBRID            — summarize old turns + keep recent turns verbatim

Each strategy has a clear tradeoff. We test all three with the SAME
conversation and show where each one fails.

Run:
    python 02_short_term_memory.py
"""

import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoder = tiktoken.encoding_for_model(MODEL)


def count_tokens(messages):
    total = 0
    for msg in messages:
        total += 4 + len(encoder.encode(msg["content"]))
    return total


def call_llm(messages, max_tokens=200):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0,
    )
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════
# THE TEST CONVERSATION
# ═══════════════════════════════════════════════════════════════
# We'll feed the same 8 turns to each strategy. The catch:
# Turn 1 has a CRITICAL FACT (user's name + food preference).
# Turn 8 asks about it. If the strategy forgot Turn 1, it fails.

SYSTEM = {"role": "system", "content": "You are a helpful assistant. Answer concisely."}

CONVERSATION = [
    "My name is Vivek. I'm allergic to peanuts — this is very important, never forget it.",
    "Tell me about the Taj Mahal.",
    "When was it built?",
    "How long did it take to construct?",
    "How many workers were involved?",
    "What's the best time to visit?",
    "Tell me about the entry ticket prices.",
    # Turn 8 — the test. Does it still remember the peanut allergy from Turn 1?
    "I'm visiting Agra next week. Can you suggest what I should eat for lunch? Remember my dietary restrictions.",
]


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: SLIDING WINDOW — Keep only the last K messages
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("STRATEGY 1: SLIDING WINDOW (keep last 4 messages)")
print("=" * 65)

print("""
  How it works:
    - Keep a full message history internally
    - But only send the last K messages to the API
    - Old messages are simply dropped

  Tradeoff:
    + Token count stays constant regardless of conversation length
    + Dead simple to implement
    - Forgets EVERYTHING older than K messages
    - No graceful degradation — facts don't fade, they vanish
""")

WINDOW_SIZE = 4  # Keep last 4 messages (2 user + 2 assistant turns)

window_messages = [SYSTEM]
full_history = [SYSTEM]

for i, user_input in enumerate(CONVERSATION, 1):
    full_history.append({"role": "user", "content": user_input})
    #full_history=["System messege","........"]

    # Build the window: system + last K messages from history (excluding system)
    conversation_only = full_history[1:]  # everything except system
    windowed = [SYSTEM] + conversation_only[-WINDOW_SIZE:]

    reply = call_llm(windowed)
    full_history.append({"role": "assistant", "content": reply})

    tokens = count_tokens(windowed)
    print(f"  Turn {i}: [{tokens:4d} tokens] You: {user_input[:55]}...")
    print(f"           AI:  {reply}")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  RESULT: The model FORGOT the peanut allergy.               │
  │  Turn 1 was dropped from the window 4 turns ago.            │
  │  For a food allergy, this is dangerous.                     │
  │                                                             │
  │  Sliding window is fine for casual chat where old context   │
  │  doesn't matter. It's TERRIBLE for anything where early     │
  │  facts are important.                                       │
  └─────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: SUMMARIZATION — Compress old turns into a summary
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("STRATEGY 2: SUMMARIZATION (compress old turns)")
print("=" * 65)

print("""
  How it works:
    - After every N turns, summarize all messages so far into a paragraph
    - Replace the old messages with the summary
    - Continue the conversation with: system + summary + new messages

  Tradeoff:
    + Retains the GIST of old conversation
    + Token count grows slowly (summary is compact)
    - Summarization loses specific details (names, numbers, allergies!)
    - Costs an extra LLM call for each summarization
""")

SUMMARIZE_EVERY = 4  # Summarize after every 4 turns

summary_messages = [SYSTEM]
running_summary = None
turns_since_summary = 0

for i, user_input in enumerate(CONVERSATION, 1):
    summary_messages.append({"role": "user", "content": user_input})

    # Build prompt: system + summary (if exists) + recent messages
    prompt = [SYSTEM]
    if running_summary:
        prompt.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{running_summary}"
        })
    # Add messages since last summary
    recent_start = len(summary_messages) - (turns_since_summary * 2 + 1)
    recent = summary_messages[max(1, recent_start):]
    prompt.extend(recent)

    reply = call_llm(prompt)
    summary_messages.append({"role": "assistant", "content": reply})
    turns_since_summary += 1

    tokens = count_tokens(prompt)
    print(f"  Turn {i}: [{tokens:4d} tokens] You: {user_input[:55]}...")

    # Time to summarize?
    if turns_since_summary >= SUMMARIZE_EVERY and i < len(CONVERSATION):
        # Summarize everything so far
        all_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in summary_messages[1:]
        )
        summarize_prompt = [
            {"role": "system", "content": "Summarize this conversation concisely. "
             "Preserve ALL specific details: names, numbers, dates, preferences, "
             "allergies, and any critical facts the user mentioned."},
            {"role": "user", "content": all_text}
        ]
        running_summary = call_llm(summarize_prompt, max_tokens=300)
        turns_since_summary = 0
        print(f"\n  [SUMMARIZED after turn {i}]")
        print(f"  Summary: {running_summary}...")
        print()

    print(f"           AI:  {reply}")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  RESULT: Check if "peanut" appears in the response.         │
  │                                                             │
  │  Sometimes the summarizer keeps it. Sometimes it doesn't.   │
  │  Summarization is LOSSY — it compresses, and compression    │
  │  loses details. You can prompt "keep all facts" but there's │
  │  no guarantee.                                              │
  │                                                             │
  │  Better than sliding window (at least it TRIES to keep      │
  │  facts), but unreliable for critical details.               │
  └─────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: HYBRID — Summary of old + recent turns verbatim
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("STRATEGY 3: HYBRID (summary + recent window)")
print("=" * 65)

print("""
  How it works:
    - Keep the last K messages verbatim (like sliding window)
    - BUT also keep a running summary of everything BEFORE the window
    - Prompt = system + summary_of_old + last_K_messages

  Tradeoff:
    + Recent context is exact (no summarization loss)
    + Old context is at least partially preserved via summary
    + Token count is bounded: summary + K messages
    - Still lossy for old specific facts (summary might drop them)
    - This is what most production chatbots actually use
""")

HYBRID_WINDOW = 4
hybrid_messages = [SYSTEM]
hybrid_summary = None

for i, user_input in enumerate(CONVERSATION, 1):
    hybrid_messages.append({"role": "user", "content": user_input})

    # Build prompt
    prompt = [SYSTEM]
    if hybrid_summary:
        prompt.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{hybrid_summary}"
        })
    # Last K messages from conversation (excluding system)
    convo = hybrid_messages[1:]
    recent = convo[-HYBRID_WINDOW:]
    prompt.extend(recent)

    reply = call_llm(prompt)
    hybrid_messages.append({"role": "assistant", "content": reply})

    tokens = count_tokens(prompt)
    print(f"  Turn {i}: [{tokens:4d} tokens] You: {user_input[:55]}...")

    # Summarize when conversation exceeds window
    convo = hybrid_messages[1:]
    if len(convo) > HYBRID_WINDOW + 2:  # +2 to trigger after window is full
        old_messages = convo[:-HYBRID_WINDOW]
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)

        context = ""
        if hybrid_summary:
            context = f"Previous summary: {hybrid_summary}\n\nNew messages:\n"

        summarize_prompt = [
            {"role": "system", "content": "Create a concise summary incorporating the previous "
             "summary (if any) and new messages. PRESERVE ALL specific details: names, numbers, "
             "allergies, preferences, and critical facts."},
            {"role": "user", "content": context + old_text}
        ]
        hybrid_summary = call_llm(summarize_prompt, max_tokens=300)

        print(f"           AI:  {reply}")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  RESULT: Hybrid is the BEST of the three for short-term     │
  │  memory. Recent turns are exact; old turns are summarized.  │
  │                                                             │
  │  But it's STILL not reliable for critical facts from early  │
  │  in the conversation. The summary MIGHT keep "peanut        │
  │  allergy" or it might compress it away.                     │
  │                                                             │
  │  This is the strategy most production chatbots use.         │
  │  ChatGPT, Claude, Gemini — all do some variant of this.     │
  └─────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# COMPARISON
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("COMPARISON: Which strategy for which use case?")
print("=" * 65)

print(f"""
  ┌──────────────────┬─────────────┬───────────────┬────────────────┐
  │                  │ Sliding     │ Summarization │ Hybrid         │
  │                  │ Window      │               │                │
  ├──────────────────┼─────────────┼───────────────┼────────────────┤
  │ Token cost       │ Constant    │ Slow growth   │ Bounded        │
  │ Old facts kept?  │ NO          │ Maybe         │ Maybe          │
  │ Recent accuracy  │ Perfect     │ Perfect       │ Perfect        │
  │ Implementation   │ Trivial     │ Medium        │ Medium         │
  │ Extra LLM calls  │ 0           │ Many          │ Some           │
  │ Best for         │ Casual chat │ Long sessions │ Production     │
  ├──────────────────┴─────────────┴───────────────┴────────────────┤
  │                                                                 │
  │  NONE of these reliably preserve specific facts from early in   │
  │  the conversation. For that, we need LONG-TERM MEMORY —         │
  │  extracting facts into a separate store. That's next.           │
  └─────────────────────────────────────────────────────────────────┘
""")