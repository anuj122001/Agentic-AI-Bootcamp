"""
Step 1: Single Tool — Code Execution

The simplest possible code execution agent. One tool: execute_python.
The agent receives a question, writes Python code, and executes it in
an E2B cloud sandbox. That's it — no data files yet, just raw code execution.

This is the core of how ChatGPT's "Code Interpreter" works:
  User asks question → LLM writes code → Sandbox runs it → Result returned

Test queries:
  - "What is the factorial of 20?"
  - "Generate the first 15 Fibonacci numbers"
  - "What day of the week was January 26, 1950?"

Dependencies:
    pip install openai e2b-code-interpreter python-dotenv

Run:
    python step1_single_tool.py
"""

import os
import json
from openai import OpenAI
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# E2B SANDBOX SETUP
# ─────────────────────────────────────────────

def create_sandbox():
    """Create a fresh E2B sandbox for code execution."""
    return Sandbox.create()


# ─────────────────────────────────────────────
# TOOL: execute_python
# ─────────────────────────────────────────────

def execute_python(sandbox, code):
    """Execute Python code in the E2B sandbox. Returns stdout or error."""
    print(f"\n  [CODE TO EXECUTE]")
    for line in code.split("\n"):
        print(f"    {line}")
    print()

    execution = sandbox.run_code(code)

    if execution.error:
        result = f"ERROR: {execution.error.name}: {execution.error.value}\n{execution.error.traceback}"
        print(f"  [RESULT] Error!\n    {execution.error.name}: {execution.error.value}")
    else:
        stdout = "\n".join(execution.logs.stdout).strip()
        result = stdout if stdout else "(no output)"
        print(f"  [RESULT] {result}")

    return result


# ─────────────────────────────────────────────
# TOOL DEFINITION (OpenAI format)
# ─────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code in a sandboxed environment. "
                           "Use this to perform calculations, data analysis, or any computation. "
                           "The code should use print() to output results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Must use print() for output."
                    }
                },
                "required": ["code"]
            }
        }
    }
]


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant. You have access to a tool called execute_python. You can use this tool to execute Python code.
Whenever user asks a question that requires mathematical calculations, you should solve that
by writing a correct and efficient Python code, you can then call execute_python tool to execute that code.

Rules:
- ALWAYS use the execute_python tool to run code — never just describe what you'd do
- Use print() to output your results
- After getting the result, explain it clearly to the user in plain English
- Keep code concise and focused on answering the question
"""


# ─────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────

def run_agent(user_query, sandbox):
    """Single-turn agent: take query, write code, execute, explain."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    print(f"\n  [AGENT] Thinking...")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=1000,
        )

        choice = response.choices[0]

        # If the model wants to call a tool
        if choice.finish_reason == "tool_calls" or choice.message.tool_calls:
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                if name == "execute_python":
                    result = execute_python(sandbox, args["code"])
                else:
                    result = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

            # Continue the loop — model might want to call another tool or respond
            continue

        # Model is done — return the final text response
        return choice.message.content


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("   STEP 1: Code Execution Agent — Single Tool")
    print("=" * 60)
    print("""
  The simplest code execution agent. You ask a question,
  the LLM writes Python code, executes it in an E2B sandbox,
  and explains the result.

  This is how ChatGPT's Code Interpreter works under the hood.

  Type 'quit' to exit.
""")

    sandbox = create_sandbox()
    print("  [SANDBOX] Created E2B sandbox\n")

    try:
        while True:
            query = input("  You: ").strip()
            if not query:
                continue
            if query.lower() == "quit":
                break

            answer = run_agent(query, sandbox)
            print(f"\n  Agent: {answer}\n")

    finally:
        sandbox.kill()
        print("\n  [SANDBOX] Closed")