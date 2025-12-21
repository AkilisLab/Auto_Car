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
                logging.exception("manual_control raised an exception")
            continue

        # placeholders for other modes
        if mode == "auto":
            from auto_mode import AutoModeController
            from path_planning.Astar_planner import (
                astar_with_intersections,
                intersection_actions,
                read_grid,
                waypoints_with_headings,
            )

            grid_path_prompt = "Enter grid file path [path_planning/test_grid.txt]: "
            grid_path = input(grid_path_prompt).strip() or "path_planning/test_grid.txt"

            try:
                grid = read_grid(grid_path)
            except Exception:
                logging.exception("Failed to read grid file '%s'", grid_path)
                print("Unable to start auto mode without a valid grid.")
                continue

            rows, cols = len(grid), len(grid[0])

            def first_walkable() -> tuple[int, int]:
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] == 0:
                            return r, c
                raise ValueError("No walkable cell found for default start")

            def last_walkable() -> tuple[int, int]:
                for r in range(rows - 1, -1, -1):
                    for c in range(cols - 1, -1, -1):
                        if grid[r][c] == 0:
                            return r, c
                raise ValueError("No walkable cell found for default goal")

            default_start = first_walkable()
            default_goal = last_walkable()

            def parse_node(raw: str, fallback: tuple[int, int]) -> tuple[int, int]:
                if not raw:
                    return fallback
                parts = [part.strip() for part in raw.split(",")]
                if len(parts) != 2:
                    raise ValueError(f"Invalid node format '{raw}'. Expected row,col")
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError as exc:
                    raise ValueError(f"Invalid integer in node '{raw}'") from exc

            try:
                start_input = input(
                    f"Enter start node row,col [{default_start[0]},{default_start[1]}]: "
                ).strip()
                goal_input = input(
                    f"Enter goal node row,col [{default_goal[0]},{default_goal[1]}]: "
                ).strip()
                start = parse_node(start_input, default_start)
                goal = parse_node(goal_input, default_goal)
            except ValueError as exc:
                logging.exception("Invalid start/goal input: %s", exc)
                print("Invalid start or goal; aborting auto mode setup.")
                continue

            try:
                path, intersections = astar_with_intersections(grid, start, goal)
            except Exception:
                logging.exception("Planner failed for start %s and goal %s", start, goal)
                print("Planner error; cannot start auto mode.")
                continue

            if not path:
                logging.warning("Planner returned empty path for start %s and goal %s", start, goal)
                print("Planner could not find a path; choose different start/goal.")
                continue

            waypoints = waypoints_with_headings(path)
            actions = intersection_actions(waypoints, intersections)
            planner_summary = {
                "grid_file": grid_path,
                "start": start,
                "goal": goal,
                "path_length": len(path),
                "intersection_indices": intersections,
                "actions": actions,
            }
            logging.info("Planner summary: %s", planner_summary)
            print(
                f"Planner ready: {planner_summary['path_length']} nodes, "
                f"{len(intersections)} intersections, actions={actions}"
            )

            planned_route = {
                "path": path,
                "waypoints": waypoints,
                "intersections": intersections,
                "actions": actions,
            }

            print("Starting auto mode with live camera")
            controller = AutoModeController(show_debug=True)
            controller.planned_route = planned_route
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
