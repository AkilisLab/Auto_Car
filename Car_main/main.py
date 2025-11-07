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
            # Launch the lane follower using the camera as input (index 0).
            # The old candidate-file logic is kept below as commented lines
            # for reference.
            import os
            from auto_mode import LaneFollower

            # candidates = [
            #     'rec_20251024_203749.avi',
            #     os.path.join('recordings', 'rec_20251024_203749.avi'),
            # ]
            # found = None
            # for p in candidates:
            #     if os.path.exists(p):
            #         found = p
            #         break
            #
            # if not found:
            #     print('Auto mode: test recording not found in expected locations:')
            #     for c in candidates:
            #         print('  -', c)
            #     print('Please place the recording in the repository root or recordings/ and try again.')
            #     return

            print("Starting auto lane follower on camera (index 0)")
            # Construct LaneFollower with the video source and then run it.
            lf = LaneFollower(video_source=0)
            # Show debug so you can see the tracking visual; set to False if headless
            try:
                lf.run(show_debug=True)
            except Exception:
                import logging

                logging.exception("LaneFollower.run raised an exception")
            # after auto mode returns, go back to chooser
            continue

        if mode == "audio":
            print("Audio mode is not implemented yet.")
            continue


if __name__ == "__main__":
    main()
