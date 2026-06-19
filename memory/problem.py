"""
Script 1: The Memory Problem — Why LLMs Forget, and Why the Naive Fix Breaks

LLMs are stateless. Every API call starts fresh — the model has ZERO memory
of anything you said before. This script demonstrates:

  1. THE AMNESIA    — tell the model your name, ask it back → it forgot
  2. THE NAIVE FIX  — append all messages to a list, send the full list each time
  3. WHY IT BREAKS  — simulate a long conversation and watch token count explode

The naive fix (send everything) works for 5 turns. It fails at 50.
This motivates every memory strategy we build in the rest of this module.

Run:
    python 01_the_problem.py
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
    """Count approximate tokens in a message list."""
    total = 0
    for msg in messages:
        # ~4 tokens overhead per message for role + formatting
        total += 4
        total += len(encoder.encode(msg["content"]))
    return total


# ═══════════════════════════════════════════════════════════════
# PART 1: THE AMNESIA — LLMs have no memory between calls
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: THE AMNESIA")
print("=" * 60)

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}

# Call 1: Tell the model something about yourself
response1 = client.chat.completions.create(
    model=MODEL,
    messages=[
        SYSTEM,
        {"role": "user", "content": "My name is Vivek. I live in Delhi and I love biryani."}
    ],
    max_tokens=100,
)
print(f"\nYou:  My name is Vivek. I live in Delhi and I love biryani.")
print(f"AI:   {response1.choices[0].message.content}")

# Call 2: Ask the model what it knows — COMPLETELY SEPARATE API call
response2 = client.chat.completions.create(
    model=MODEL,
    messages=[
        SYSTEM,
        {"role": "user", "content": "What is my name and what food do I like?"}
    ],
    max_tokens=100,
)
print(f"\nYou:  What is my name and what food do I like?")
print(f"AI:   {response2.choices[0].message.content}")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  The model FORGOT. Each API call is a clean slate.      │
  │  Call 2 has no idea what happened in Call 1.             │
  │  This is not a bug — it's how the API works.            │
  │  The model is a FUNCTION: input → output. No state.     │
  └─────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# PART 2: THE NAIVE FIX — Append all messages, send everything
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 2: THE NAIVE FIX — send full conversation history")
print("=" * 60)

messages = [SYSTEM]


def chat(user_input):
    """Send full message history + new input, append response."""
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=150,
    )

    assistant_msg = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_msg})
    return assistant_msg


# A short conversation — the naive fix works perfectly here
conversation = [
    "My name is Vivek. I live in Delhi and I love biryani.",
    "I'm a software engineer. I work at a startup.",
    "What is my name and where do I live?",
    "What do I do for work?",
    "What food do I like?",
]

for user_input in conversation:
    reply = chat(user_input)
    tokens = count_tokens(messages)
    print(f"\n  You:    {user_input}")
    print(f"  AI:     {reply}")
    print(f"  Tokens: {tokens} (in the full message list sent to API)")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  It remembers! The trick: we send ALL prior messages    │
  │  in every API call. The model "remembers" because the   │
  │  old messages are right there in the prompt.            │
  │                                                         │
  │  This is SHORT-TERM MEMORY — it's just the messages     │
  │  list. Nothing fancy. But it has a ceiling...           │
  └─────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════
# PART 3: WHY IT BREAKS — Simulate a long conversation
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 3: WHY THE NAIVE FIX BREAKS AT SCALE")
print("=" * 60)

# Simulate a realistic 30-turn conversation without making API calls.
# Each turn: user says ~30 words, assistant replies ~60 words.
# We'll just count how the tokens grow.

simulated_messages = [SYSTEM]

# Realistic conversation turns (no actual API calls — just measuring growth)
turns = [
    ("Tell me about the history of Delhi", "Delhi, one of the oldest cities in the world, has been continuously inhabited since the 6th century BCE. It has served as the capital of several empires including the Delhi Sultanate and the Mughal Empire. The city was designed by Edwin Lutyens as New Delhi in 1911 when the British moved the capital from Calcutta."),
    ("What about the food scene there?", "Delhi is famous for its street food culture. Chandni Chowk is the epicenter — you'll find legendary paranthe wali gali, Karim's kebabs dating back to 1913, and Natraj's chaat. Old Delhi specializes in Mughlai cuisine while South Delhi has modern cafes and international restaurants. The city's butter chicken was invented here at Moti Mahal."),
    ("How's the weather throughout the year?", "Delhi experiences extreme seasons. Summers (April-June) hit 45-48°C regularly. Monsoon (July-September) brings heavy rainfall and humidity. Winters (December-February) drop to 2-4°C with dense fog. October-November and February-March are the pleasant windows. Air quality deteriorates significantly in November due to crop burning in neighboring states."),
    ("What are the best areas to live in?", "For professionals, Gurgaon and Noida offer modern apartments near tech hubs. South Delhi (GK, Hauz Khas, Vasant Kunj) is upscale with good restaurants. Dwarka is affordable and well-connected via metro. Connaught Place is central but expensive. Rohini in North Delhi offers good infrastructure at moderate prices. Your choice depends on workplace proximity and budget."),
    ("Tell me about the metro system", "Delhi Metro is India's largest rapid transit system with 286 stations across 12 lines spanning 393 km. It carries 5-6 million passengers daily. The Airport Express connects New Delhi station to IGI Airport in 20 minutes. Smart cards and QR tickets are available. It has dramatically reduced road congestion and is expanding with Phase IV adding 4 new corridors."),
    ("What about the startup ecosystem?", "Delhi-NCR is India's largest startup ecosystem by funding — $15B+ raised in 2023 alone. Key hubs: Gurgaon (fintech, SaaS), Noida (edtech, IT services), and Delhi (D2C, food-tech). Major unicorns include Zomato, PolicyBazaar, Delhivery, and PhysicsWallah. Coworking spaces like WeWork, 91springboard are everywhere. The IIT Delhi campus is a major talent pipeline."),
    ("Compare Delhi vs Bangalore for tech jobs", "Bangalore has more pure tech companies and a mature engineering culture — Google, Microsoft, Amazon all have large offices. Delhi-NCR has more diverse roles — consulting, fintech, e-commerce, media. Bangalore salaries for senior engineers are slightly higher but so is rent. Delhi has better infrastructure (metro vs Bangalore traffic). Delhi-NCR's advantage: proximity to government and enterprise clients."),
    ("What's the cost of living?", "A 2BHK in a decent Gurgaon society: Rs 25-35K/month. South Delhi: Rs 35-50K. Utilities: Rs 3-5K. Groceries for a couple: Rs 8-12K. Eating out budget: Rs 5-10K. A meal at a mid-range restaurant: Rs 800-1200 for two. Auto/cab commute: Rs 3-5K if not using metro. Total comfortable living for a couple: Rs 60-80K/month excluding rent."),
    ("Tell me about the historical monuments", "Delhi has three UNESCO World Heritage Sites: Qutub Minar (1193), Humayun's Tomb (1570), and Red Fort (1648). Other must-visits: India Gate, Lotus Temple, Jama Masjid, Purana Qila, Jantar Mantar, and Akshardham Temple. Mehrauli Archaeological Park has ruins spanning 1000 years. Tughlaqabad Fort is an underrated gem. Most monuments are maintained by ASI with entry fees of Rs 20-40 for Indians."),
    ("What about education options?", "Delhi has India's top institutions: IIT Delhi, JNU, Delhi University, AIIMS, NLU, IIM (in nearby Noida). DU's North Campus is iconic for undergraduate education. Coaching hubs in Mukherjee Nagar (UPSC) and Rajinder Nagar (banking/SSC) attract students nationwide. International schools like American Embassy, British School charge Rs 10-20 lakh/year. The city produces the most UPSC toppers annually."),
]

# We only have 10 detailed turns above but let's duplicate/extend to 30
# to show the scaling problem
extended_turns = turns * 3  # 30 turns total

print(f"\n  Simulating a 30-turn conversation (no API calls, just counting tokens):\n")

COST_PER_1M_INPUT = 0.15   # gpt-4o-mini input pricing
COST_PER_1M_OUTPUT = 0.60  # gpt-4o-mini output pricing

total_input_tokens = 0

for i, (user_msg, ai_msg) in enumerate(extended_turns, 1):
    simulated_messages.append({"role": "user", "content": user_msg})
    simulated_messages.append({"role": "assistant", "content": ai_msg})

    tokens_in_history = count_tokens(simulated_messages)
    # Each API call sends the FULL history as input
    total_input_tokens += tokens_in_history
    cost_so_far = (total_input_tokens / 1_000_000) * COST_PER_1M_INPUT

    if i in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  Turn {i:2d}: {tokens_in_history:,} tokens per call | "
              f"Cumulative input tokens: {total_input_tokens:,} | "
              f"Cumulative cost: ${cost_so_far:.4f}")

print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  WHAT JUST HAPPENED:                                    │
  │                                                         │
  │  Turn  1:  ~200 tokens per call                         │
  │  Turn 30: ~6000+ tokens per call                        │
  │                                                         │
  │  Every turn, the input grows. You're paying to re-send  │
  │  the ENTIRE conversation history every single time.     │
  │                                                         │
  │  THREE PROBLEMS:                                        │
  │                                                         │
  │  1. COST — you pay for all prior turns as input tokens  │
  │     on every single call. Turn 30 pays for all 30 turns │
  │     of context. Costs grow quadratically (1+2+3+...+N). │
  │                                                         │
  │  2. CONTEXT WINDOW — models have limits (128K tokens).  │
  │     A busy chat app with 200 turns? You hit the wall.   │
  │                                                         │
  │  3. ATTENTION DILUTION — the model has to attend to     │
  │     everything. Turn 1's details get "buried" under 29  │
  │     turns of other context. Quality degrades.           │
  │     (This is "context rot" — we saw it in the webinar.) │
  │                                                         │
  │  NEXT STEP: smarter strategies for managing memory.     │
  └─────────────────────────────────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════
# PART 4: QUICK COMPARISON — What we'll build in this module
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("THE MEMORY STRATEGIES WE'LL BUILD")
print("=" * 60)

print(f"""
  ┌──────────────────────┬────────────────────────────────────────┐
  │ Strategy             │ How it works                           │
  ├──────────────────────┼────────────────────────────────────────┤
  │ Naive (this script)  │ Send everything. Breaks at scale.      │
  │                      │                                        │
  │ Sliding Window       │ Keep only the last K messages.         │
  │                      │ Forgets old context entirely.          │
  │                      │                                        │
  │ Summarization        │ Summarize old turns into a paragraph.  │
  │                      │ Saves tokens but loses specific facts. │
  │                      │                                        │
  │ Fact Extraction      │ LLM extracts atomic facts ("user is    │
  │                      │ vegetarian", "works at Google") into a  │
  │                      │ separate store. Retrieves on demand.   │
  │                      │                                        │
  │ Mem0 (production)    │ Does extraction + dedup + contradiction │
  │                      │ resolution + vector search. One library.│
  │                      │                                        │
  │ Memory as Tool       │ Agent DECIDES what to remember via     │
  │                      │ tool calls. Conscious memory, not      │
  │                      │ passive logging.                       │
  └──────────────────────┴────────────────────────────────────────┘

  Each strategy has tradeoffs. No silver bullet.
  Let's build each one and see where they shine and fail.
""")