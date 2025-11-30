"""A* path planner using Manhattan distance heuristic.

This module reads a grid from a text file where each character is either
0 (walkable) or 1 (wall). It implements the A* search algorithm (4-connected
grid) with Manhattan distance as the heuristic and returns the path as a list
of nodes (row, col) from start to goal (inclusive).

File format (example):
00010
00110
00000

Whitespace and newlines are ignored; lines should be equal length.

Functions:
 - read_grid(file_path) -> List[List[int]]
 - astar(grid, start, goal) -> List[Tuple[int,int]]
 - waypoints_with_headings(path) -> List[dict]
 - plan_with_headings(grid, start, goal) -> List[dict]

The module also exposes a small CLI for quick testing.

Author: generated for user. Please review comments and adjust start/goal
handling or connectivity if you need diagonal moves.
"""

from typing import List, Tuple, Dict, Optional
import heapq
import sys
import os

Grid = List[List[int]]
Node = Tuple[int, int]


def read_grid(file_path: str) -> Grid:
	"""Read a grid from a text file and return a 2D list of ints.

	Each line corresponds to a row. Characters '0' and '1' are respected.
	Other whitespace is ignored. Lines must be the same length.
	"""
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"Grid file not found: {file_path}")

	grid: Grid = []
	with open(file_path, 'r') as f:
		for raw in f:
			line = raw.strip()
			if not line:
				continue
			row: List[int] = []
			for ch in line:
				if ch == '0':
					row.append(0)
				elif ch == '1':
					row.append(1)
				else:
					# allow space separated numbers as well
					if ch.isspace():
						continue
					raise ValueError(f"Invalid character '{ch}' in grid file")
			grid.append(row)

	if not grid:
		raise ValueError("Grid file is empty or contains only blank lines")

	# Ensure rectangular
	width = len(grid[0])
	for r, row in enumerate(grid):
		if len(row) != width:
			raise ValueError(f"Inconsistent row length at row {r}: {len(row)} vs {width}")

	return grid


def manhattan(a: Node, b: Node) -> int:
	"""Manhattan distance between two nodes."""
	return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(grid: Grid, node: Node) -> List[Node]:
	"""Return walkable 4-connected neighbors of node within grid bounds."""
	rows = len(grid)
	cols = len(grid[0])
	r, c = node
	nbrs: List[Node] = []
	for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
		nr, nc = r + dr, c + dc
		if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
			nbrs.append((nr, nc))
	return nbrs


def reconstruct_path(came_from: Dict[Node, Node], current: Node) -> List[Node]:
	"""Reconstruct path from start to current using came_from map."""
	path: List[Node] = [current]
	while current in came_from:
		current = came_from[current]
		path.append(current)
	path.reverse()
	return path


def astar(grid: Grid, start: Node, goal: Node) -> List[Node]:
	"""Run A* on the given grid from start to goal.

	Returns a list of nodes (row, col) from start to goal inclusive.
	If no path exists, returns an empty list.

	Assumptions/contract:
	- grid: 0 is walkable, 1 is obstacle
	- start and goal are inside bounds
	- uses 4-connectivity and Manhattan heuristic
	"""
	rows = len(grid)
	cols = len(grid[0])
	sr, sc = start
	gr, gc = goal

	# Basic bounds checks
	if not (0 <= sr < rows and 0 <= sc < cols):
		raise ValueError("Start node out of grid bounds")
	if not (0 <= gr < rows and 0 <= gc < cols):
		raise ValueError("Goal node out of grid bounds")
	if grid[sr][sc] != 0:
		raise ValueError("Start node is not walkable")
	if grid[gr][gc] != 0:
		raise ValueError("Goal node is not walkable")

	open_heap: List[Tuple[int, int, Node]] = []  # (f_score, tie_breaker, node)
	g_score: Dict[Node, int] = {start: 0}
	f_score: Dict[Node, int] = {start: manhattan(start, goal)}
	came_from: Dict[Node, Node] = {}

	# tie breaker counter to ensure deterministic behavior
	counter = 0
	heapq.heappush(open_heap, (f_score[start], counter, start))

	closed: set[Node] = set()

	while open_heap:
		_, _, current = heapq.heappop(open_heap)

		if current == goal:
			return reconstruct_path(came_from, current)

		if current in closed:
			continue
		closed.add(current)

		current_g = g_score.get(current, float('inf'))
		for nb in neighbors(grid, current):
			if nb in closed:
				continue
			tentative_g = current_g + 1  # uniform cost per move

			if tentative_g < g_score.get(nb, float('inf')):
				came_from[nb] = current
				g_score[nb] = tentative_g
				f = tentative_g + manhattan(nb, goal)
				f_score[nb] = f
				counter += 1
				heapq.heappush(open_heap, (f, counter, nb))

	# no path found
	return []


def _delta_to_heading(dr: int, dc: int) -> str:
	"""Map grid step delta to a cardinal heading string.

	Conventions (row increases downwards):
	- (-1, 0) -> 'N' (up)
	- ( 1, 0) -> 'S' (down)
	- ( 0,-1) -> 'W' (left)
	- ( 0, 1) -> 'E' (right)
	"""
	if dr == -1 and dc == 0:
		return 'N'
	if dr == 1 and dc == 0:
		return 'S'
	if dr == 0 and dc == -1:
		return 'W'
	if dr == 0 and dc == 1:
		return 'E'
	# Should not happen for 4-connected paths; fallback
	return 'E'


def waypoints_with_headings(path: List[Node]) -> List[dict]:
	"""Convert a list of grid nodes into waypoints each with heading.

	Returns a list of dicts: { 'row': r, 'col': c, 'heading': 'N'|'S'|'E'|'W' }.
	For the final waypoint, the heading is copied from the last move if available.
	"""
	if not path:
		return []
	n = len(path)
	waypoints: List[dict] = []
	last_heading = 'E'
	for i in range(n):
		r, c = path[i]
		if i < n - 1:
			rn, cn = path[i + 1]
			dr, dc = rn - r, cn - c
			hd = _delta_to_heading(dr, dc)
			last_heading = hd
		else:
			hd = last_heading
		waypoints.append({'row': r, 'col': c, 'heading': hd})
	return waypoints


def plan_with_headings(grid: Grid, start: Node, goal: Node) -> List[dict]:
	"""Run A* and return waypoints with headings.

	This is a convenience wrapper around astar()+waypoints_with_headings().
	"""
	path = astar(grid, start, goal)
	return waypoints_with_headings(path)


if __name__ == '__main__':
	# Simple command-line interface for quick tests.
	# Usage examples:
	# python Astar_planner.py grid.txt "0,0" "4,3"
	# If start/goal not provided, defaults to top-left and bottom-right walkable cells.
	import argparse

	parser = argparse.ArgumentParser(description='A* planner (Manhattan) on a text grid')
	parser.add_argument('grid_file', help='Path to grid text file')
	parser.add_argument('--start', help='Start as r,c (zero-indexed)', default=None)
	parser.add_argument('--goal', help='Goal as r,c (zero-indexed)', default=None)
	args = parser.parse_args()

	grid = read_grid(args.grid_file)

	rows = len(grid)
	cols = len(grid[0])

	def parse_node(s: Optional[str], default: Node) -> Node:
		if s is None:
			return default
		try:
			parts = s.split(',')
			if len(parts) != 2:
				raise ValueError
			return (int(parts[0]), int(parts[1]))
		except Exception:
			raise ValueError(f"Invalid node format: {s}. Expected r,c")

	default_start = (0, 0)
	if grid[0][0] != 0:
		# find first walkable cell for default
		found = False
		for r in range(rows):
			for c in range(cols):
				if grid[r][c] == 0:
					default_start = (r, c)
					found = True
					break
			if found:
				break

	default_goal = (rows - 1, cols - 1)
	if grid[default_goal[0]][default_goal[1]] != 0:
		# find last walkable cell for default
		found = False
		for r in range(rows - 1, -1, -1):
			for c in range(cols - 1, -1, -1):
				if grid[r][c] == 0:
					default_goal = (r, c)
					found = True
					break
			if found:
				break

	start = parse_node(args.start, default_start)
	goal = parse_node(args.goal, default_goal)

	print(f"Grid size: {rows}x{cols}")
	print(f"Start: {start}, Goal: {goal}")

	path = astar(grid, start, goal)
	if path:
		print("Path found (length {}):".format(len(path)))
		print(path)
		# Also show waypoints with headings
		wps = waypoints_with_headings(path)
		print("Waypoints with headings:")
		for wp in wps:
			print(wp)
	else:
		print("No path found")

