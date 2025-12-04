from __future__ import annotations
from graph import run_agent

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(prog="nova-assistant", description="Interactive CLI for the NovaCRM agent.")
    parser.add_argument("--account", "--account-id", dest="account", default=None, help="Optional account identifier for data lookups.")
    parser.add_argument("--temp", type=float, default=0.0, help="LLM sampling temperature (default: 0.0).")

    return parser.parse_args()


def pretty_print_response(result: dict) -> None:

    answer = result.get("answer") or "No answer."
    evidence = result.get("evidence") or []
    errors = result.get("errors") or []
    confidence = result.get("confidence")

    print("\n--- Assistant Response ---\n")
    print(answer)

    if evidence:
        print("\nEvidence:")
        for item in evidence:
            print(f"- {item}")

    if errors:
        print("\nNotes:")
        for err in errors:
            print(f"- {err}")

    if confidence is not None:
        print(f"\n(Internal confidence: {confidence})")

    print("\n--------------------------\n")


def run_cli() -> None:

    args = parse_args()
    print("NovaCRM CLI Assistant. Type 'exit' or 'quit' to leave.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting NovaCRM CLI Assistant.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = run_agent(user_input, account=args.account, temperature=args.temp)
        
        pretty_print_response(result)


if __name__ == "__main__":
    run_cli()
