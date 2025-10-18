#!/usr/bin/env python3
"""Top-level launcher for Auto_Car control modes.

Prompts the user to choose a control mode. Currently only 'manual'
is implemented and will call into `manual_rc.manual_control()`.
"""

from __future__ import annotations

import sys


def choose_mode() -> str:
    choices = {
        "m": "manual",
        "a": "auto",
        "u": "audio",
        "q": "quit",
    }
    prompt = (
        "Choose control mode:\n"
        "  (m) manual\n"
        "  (a) auto [not implemented]\n"
        "  (u) audio [not implemented]\n"
        "  (q) quit\n"
        "Enter choice: "
    )
    while True:
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if not choice:
            continue
        key = choice[0]
        if key in choices:
            return choices[key]
        print("Invalid choice, try again.")


def main() -> None:
    mode = choose_mode()
    if mode == "quit":
        print("Exiting.")
        return

    if mode == "manual":
        # Import lazily so the CLI can run even if manual dependencies
        # are missing until this branch is chosen.
        from manual_rc import manual_control

        print("Starting manual control. Press 'p' to exit the manual controller.")
        manual_control()
        return

    # placeholders for other modes
    if mode == "auto":
        print("Auto mode is not implemented yet.")
    elif mode == "audio":
        print("Audio mode is not implemented yet.")


if __name__ == "__main__":
    main()
