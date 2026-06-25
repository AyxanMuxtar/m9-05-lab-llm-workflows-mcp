"""
Lab | Build the Loop Yourself
==============================
Hand-rolled tool-call loop using Ollama -- no API key, runs locally.

Model: change MODEL_NAME below to switch between available Ollama models.
       Options tried: gemma4:e4b  /  minimax-m3:cloud

Features:
  - messages list as short-term memory (full history sent every call)
  - step limit so the loop can never run away
  - two built-in tools: lookup_order  and  calculate
  - verbose printing so every loop step is visible
"""

import json
import os

import ollama

# ---------------------------------------------------------------------------
# 1. Model selection -- change here to switch models
# ---------------------------------------------------------------------------
# MODEL_NAME = "minimax-m3:cloud"
MODEL_NAME = "gemma4:e4b"

# ---------------------------------------------------------------------------
# 2. Load orders data
# ---------------------------------------------------------------------------
ORDERS_FILE = os.path.join(os.path.dirname(__file__), "orders.json")
with open(ORDERS_FILE, "r") as fh:
    ORDERS: dict = json.load(fh)

# ---------------------------------------------------------------------------
# 3. Tool implementations (plain Python functions)
# ---------------------------------------------------------------------------

def lookup_order(order_id: str) -> str:
    """Return order details as a JSON string, or an error message."""
    order = ORDERS.get(order_id.strip().upper())
    if order is None:
        return json.dumps({"error": f"Order {order_id!r} not found."})
    return json.dumps({**order, "order_id": order_id})


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return the result."""
    # Allow only safe characters: digits, operators, spaces, dots, parentheses
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Unsafe expression rejected."})
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return json.dumps({"result": result})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# Map name -> callable so we can dispatch by name
TOOL_FUNCTIONS = {
    "lookup_order": lookup_order,
    "calculate": calculate,
}

# ---------------------------------------------------------------------------
# 4. Tool declarations -- Ollama uses the OpenAI "tools" format
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "Look up an order from the database by its order ID "
                "(e.g. A1001). Returns item name, price, purchase date, "
                "and warranty information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up, e.g. A1001",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a simple arithmetic expression and return the result. "
                "Example: '1200 * 3' or '(100 + 50) / 2'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python arithmetic expression using +, -, *, /",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 5. The hand-rolled loop
# ---------------------------------------------------------------------------

MAX_STEPS = 5  # hard cap -- the loop will never exceed this many model calls


def run_agent(user_query: str, messages: list, *, verbose: bool = True) -> str:
    """
    Add *user_query* to *messages*, then run the model->tool->model loop.

    Returns the model's final text answer.
    The *messages* list is mutated in place so memory persists across calls.

    Ollama message format (OpenAI-compatible):
      - User turn:     {"role": "user",      "content": "..."}
      - Model turn:    {"role": "assistant",  "content": "...", "tool_calls": [...]}
      - Tool result:   {"role": "tool",       "tool_call_id": "...", "content": "..."}
    """
    # Append the new user turn to the running conversation
    messages.append({"role": "user", "content": user_query})

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER: {user_query}")
        print(f"{'='*60}")

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n--- Step {step} / {MAX_STEPS} ---")
            print(f"    Sending {len(messages)} message(s) to model [{MODEL_NAME}] ...")

        # ------------------------------------------------------------------
        # Call the model with the FULL conversation history (= short-term memory)
        # ------------------------------------------------------------------
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DECLARATIONS,
        )

        assistant_msg = response["message"]  # dict with role, content, tool_calls

        # Append the model's reply to memory (preserves tool_calls if present)
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            # No tool call -> the model produced a final text answer
            final_text = assistant_msg.get("content", "").strip()
            if verbose:
                print(f"\nMODEL (final answer): {final_text}")
            return final_text

        # ------------------------------------------------------------------
        # Execute each tool the model requested, then loop again
        # ------------------------------------------------------------------
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            # Ollama may return args as a string or a dict
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            tool_call_id = tc.get("id", fn_name)  # id may not always be present

            if verbose:
                print(f"    -> Tool call: {fn_name}({fn_args})")

            if fn_name not in TOOL_FUNCTIONS:
                result_str = json.dumps({"error": f"Unknown tool: {fn_name!r}"})
            else:
                result_str = TOOL_FUNCTIONS[fn_name](**fn_args)

            if verbose:
                print(f"    <- Tool result: {result_str}")

            # Append the tool result as a "tool" role message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_str,
                }
            )

    # If we exit the loop without a final answer, the step limit was hit
    if verbose:
        print(f"\n[!] Step limit ({MAX_STEPS}) reached -- stopping the loop.")
    return "Sorry, I could not finish in time (step limit reached)."


# ---------------------------------------------------------------------------
# 6. Main -- two-turn memory demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # This list IS the short-term memory -- it persists across both turns
    messages: list = []

    print("\n" + "=" * 60)
    print("  DEMO: Two-turn conversation with memory")
    print("=" * 60)

    # -- Turn 1 -------------------------------------------------------------
    answer1 = run_agent("What did order A1001 cost?", messages)
    print(f"\n[OK] Answer 1: {answer1}")

    # -- Turn 2 -------------------------------------------------------------
    # "three of them" is deliberately vague -- it only makes sense because
    # the model can see Turn 1 still sitting in messages.
    answer2 = run_agent("And what about three of them?", messages)
    print(f"\n[OK] Answer 2: {answer2}")

    # ------------------------------------------------------------------
    # Memory proof: dump every message so the human can see what was sent
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Memory proof: full messages list after both turns")
    print("=" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if tool_calls:
            for tc in tool_calls:
                fn = tc["function"]
                print(f"  [{i}] {role} -> TOOL CALL: {fn['name']}({fn['arguments']})")
        elif msg["role"] == "tool":
            print(f"  [{i}] {role} <- TOOL RESULT: {content[:120]}")
        else:
            print(f"  [{i}] {role}: {content[:120]}")
