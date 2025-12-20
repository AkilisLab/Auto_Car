#!/usr/bin/env python3
"""Top-level launcher for Auto_Car control modes.

Prompts the user to choose a control mode. Currently only 'manual'
is implemented and will call into `manual_rc.manual_control()`.
"""

from __future__ import annotations

import logging
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
        "  (a) auto [partially implemented]\n"
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
    # Run choose_mode repeatedly so returning from `manual_control()`
    # (when the user presses 'p') brings the user back to the mode
    # chooser instead of exiting the program.
    while True:
        mode = choose_mode()
        if mode == "quit":
            print("Exiting.")
            break

        if mode == "manual":
            # Import lazily so the CLI can run even if manual dependencies
            # are missing until this branch is chosen.
            from manual_rc import manual_control

            print("Starting manual control. Press 'p' to exit the manual controller.")
            # When manual_control() returns (user pressed 'p'), loop back
            # to the chooser to allow selecting a different mode.
            try:
                manual_control()
            except Exception:
                # If manual control raises, log and continue to chooser
                import logging

                logging.exception("manual_control raised an exception")
            continue

        # placeholders for other modes
        if mode == "auto":
            from auto_mode import AutoModeController

            print("Starting auto mode with live camera")
            controller = AutoModeController(show_debug=True)
            try:
                controller.run()
            except Exception:
                logging.exception("Auto mode failed")
            continue

        if mode == "audio":
            print("Audio mode is not implemented yet.")
            continue


if __name__ == "__main__":
    main()
