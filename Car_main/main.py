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
            # Launch the lane follower using the camera as input (index 0).
            # The old candidate-file logic is kept below as commented lines
            # for reference.
            import os
            from auto_mode import LaneFollower
            # A* path planning imports (local grid planner)
            try:
                from path_planning.Astar_planner import read_grid, plan_with_headings
            except Exception:
                read_grid = None
                plan_with_headings = None

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
            # --- New: optional path planning step ---
            if read_grid and plan_with_headings:
                grid_path_default = os.path.join('path_planning', 'test_grid.txt')
                grid_path = grid_path_default if os.path.exists(grid_path_default) else None
                if grid_path is None:
                    print(f"Grid file not found at {grid_path_default}; skipping path planning.")
                else:
                    try:
                        grid = read_grid(grid_path)
                        rows, cols = len(grid), len(grid[0])

                        def find_default_start_goal():
                            # find first free cell
                            start = None
                            for r in range(rows):
                                for c in range(cols):
                                    if grid[r][c] == 0:
                                        start = (r, c)
                                        break
                                if start:
                                    break
                            # find last free cell
                            goal = None
                            for r in range(rows - 1, -1, -1):
                                for c in range(cols - 1, -1, -1):
                                    if grid[r][c] == 0:
                                        goal = (r, c)
                                        break
                                if goal:
                                    break
                            return start, goal

                        default_start, default_goal = find_default_start_goal()
                        print(f"Grid loaded ({rows}x{cols}). Default start={default_start} goal={default_goal}")
                        # Prompt user for start/goal (row,col)
                        def parse_coord(s, default):
                            s = s.strip()
                            if not s:
                                return default
                            try:
                                parts = s.split(',')
                                if len(parts) != 2:
                                    raise ValueError
                                r, c = int(parts[0]), int(parts[1])
                                if not (0 <= r < rows and 0 <= c < cols):
                                    print("Coordinates out of bounds; using default.")
                                    return default
                                if grid[r][c] != 0:
                                    print("Cell blocked; using default.")
                                    return default
                                return (r, c)
                            except Exception:
                                print("Invalid format; using default.")
                                return default

                        try:
                            user_start = input("Enter start r,c (blank for default): ")
                        except (EOFError, KeyboardInterrupt):
                            user_start = ''
                        try:
                            user_goal = input("Enter goal r,c (blank for default): ")
                        except (EOFError, KeyboardInterrupt):
                            user_goal = ''
                        start = parse_coord(user_start, default_start)
                        goal = parse_coord(user_goal, default_goal)
                        print(f"Planning path from {start} to {goal} ...")
                        waypoints = plan_with_headings(grid, start, goal)
                        if not waypoints:
                            print("No path found; proceeding without path guidance.")
                        else:
                            print(f"Path found with {len(waypoints)} waypoints.")
                            # Show condensed waypoint list (first few + last)
                            preview = waypoints[:5]
                            if len(waypoints) > 7:
                                preview.append({'row': '...', 'col': '...', 'heading': '...'})
                            if len(waypoints) > 5:
                                preview.extend(waypoints[-2:])
                            print("Waypoints (preview):")
                            for wp in preview:
                                print(wp)
                            # Extract simple turning indices where heading changes
                            turning_points = []
                            prev_h = None
                            for idx, wp in enumerate(waypoints):
                                h = wp['heading']
                                if prev_h is not None and h != prev_h:
                                    turning_points.append(idx)
                                prev_h = h
                            print(f"Turning point indices: {turning_points if turning_points else 'None'}")
                    except Exception as e:
                        print(f"Path planning error: {e}; continuing without path.")
            else:
                print("Path planner not available (import failed); skipping path guidance.")
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
