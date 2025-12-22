"""Manual test harness for validating Raspbot turn and lane-change routines."""

import argparse
import sys
import time
from typing import Iterable, Tuple, Any

# Step tuple encodes differential drive command: (left_dir, left_speed, right_dir, right_speed, duration)
Step = Tuple[int, int, int, int, float]


FORWARD_SPEED = 30
SPIN_SPEED = 30
LANE_SHIFT_SPEED = 30
SHORT_FORWARD_TIME = 0.9
LONG_FORWARD_TIME = 2
SPIN_TIME = 0.7
ARC_TIME = 0.5
SHORT_GAP = 0.25  # pause between individual steps to ensure motors settle


def _execute_steps(bot: Any, steps: Iterable[Step]) -> None:
	"""Replay a sequence of differential commands on the Raspbot motors, with IR-based correction."""
	if bot is None:
		raise RuntimeError("Raspbot hardware unavailable; aborting test")
	CORRECTION = 12  # speed adjustment for correction
	CORR_INTERVAL = 0.05  # seconds between IR checks
	for left_dir, left_speed, right_dir, right_speed, duration in steps:
		t0 = time.time()
		while time.time() - t0 < duration:
			# Only apply correction for forward/arc (not spins)
			if left_dir == 0 and right_dir == 0:
				try:
					ir = bot.read_data_array(0x0A, 1)
					if ir and len(ir) > 0:
						track = int(ir[0])
						leftmost = (track >> 3) & 0x01
						rightmost = track & 0x01
						l_speed, r_speed = left_speed, right_speed
						# If rightmost sensor sees white, nudge left
						if rightmost:
							l_speed = max(0, left_speed - CORRECTION)
							r_speed = min(255, right_speed + CORRECTION)
						# If leftmost sensor sees white, nudge right
						elif leftmost:
							l_speed = min(255, left_speed + CORRECTION)
							r_speed = max(0, right_speed - CORRECTION)
						for motor_id in (0, 1):
							bot.Ctrl_Car(motor_id, left_dir, l_speed)
						for motor_id in (2, 3):
							bot.Ctrl_Car(motor_id, right_dir, r_speed)
					else:
						# Fallback to default speeds if IR read fails
						for motor_id in (0, 1):
							bot.Ctrl_Car(motor_id, left_dir, left_speed)
						for motor_id in (2, 3):
							bot.Ctrl_Car(motor_id, right_dir, right_speed)
				except Exception:
					# On error, fallback to default speeds
					for motor_id in (0, 1):
						bot.Ctrl_Car(motor_id, left_dir, left_speed)
					for motor_id in (2, 3):
						bot.Ctrl_Car(motor_id, right_dir, right_speed)
				time.sleep(CORR_INTERVAL)
			else:
				# For spins, just run as before
				for motor_id in (0, 1):
					bot.Ctrl_Car(motor_id, left_dir, left_speed)
				for motor_id in (2, 3):
					bot.Ctrl_Car(motor_id, right_dir, right_speed)
				time.sleep(duration)
				break
		# brief stop to allow the car to settle between maneuvers
		_stop(bot)
		time.sleep(SHORT_GAP)
	_stop(bot)


def _stop(bot: Any) -> None:
	"""Stop all motors (safe fallback after scripted motion)."""
	for motor_id in range(4):
		bot.Ctrl_Car(motor_id, 0, 0)


def _forward(duration: float) -> Step:
	"""Drive straight ahead at nominal cruise speed for `duration` seconds."""
	return (0, FORWARD_SPEED*1.1, 0, FORWARD_SPEED, duration)


def _forward_slow(duration: float) -> Step:
	"""Drive forward at lower speed to smooth lane-change transitions."""
	return (0, LANE_SHIFT_SPEED*1.1, 0, LANE_SHIFT_SPEED, duration)


def _spin_left(duration: float) -> Step:
	"""Pivot left in place using opposing wheel directions."""
	return (1, SPIN_SPEED, 0, SPIN_SPEED, duration)


def _spin_right(duration: float) -> Step:
	"""Pivot right in place using opposing wheel directions."""
	return (0, SPIN_SPEED, 1, SPIN_SPEED, duration)


def _arc_left(duration: float) -> Step:
	"""Perform a gentle left arc for lane changes."""
	return (0, LANE_SHIFT_SPEED // 2, 0, LANE_SHIFT_SPEED, duration)


def _arc_right(duration: float) -> Step:
	"""Perform a gentle right arc for lane changes."""
	return (0, LANE_SHIFT_SPEED, 0, LANE_SHIFT_SPEED // 2, duration)


ACTION_STEPS = {
	"left_type_a": (
		_forward(SHORT_FORWARD_TIME*1.7),
		_spin_left(SPIN_TIME),
		# _forward(SHORT_FORWARD_TIME*1.3),
	),
	"left_type_b": (
		_forward(SHORT_FORWARD_TIME*1.7),
		_spin_left(SPIN_TIME),
		# _forward(LONG_FORWARD_TIME),
	),
	"left_type_c": (
		_forward(LONG_FORWARD_TIME),
		_spin_left(SPIN_TIME),
		# _forward(SHORT_FORWARD_TIME),
	),
	"right": (
		_forward(SHORT_FORWARD_TIME),
		_spin_right(SPIN_TIME),
		# _forward(SHORT_FORWARD_TIME),
	),
	"forward": (
		_forward(SHORT_FORWARD_TIME*2),
	),
	"lane_change_left": (
		_forward_slow(SHORT_FORWARD_TIME),
		_arc_left(ARC_TIME),
		_forward_slow(SHORT_FORWARD_TIME),
		_arc_right(ARC_TIME),
		# _forward_slow(SHORT_FORWARD_TIME),
	),
	"lane_change_right": (
		_forward_slow(SHORT_FORWARD_TIME),
		_arc_right(ARC_TIME),
		_forward_slow(SHORT_FORWARD_TIME),
		_arc_left(ARC_TIME),
		# _forward_slow(SHORT_FORWARD_TIME),
	),
}


def run_action(action_name: str, bot: Any) -> None:
	"""Play back the scripted maneuver on the provided `bot` controller.

	The caller (for example `AutoModeController`) should pass the Raspbot
	controller instance (available via `drive.get_controller()`). This
	function will raise if `bot` is None.
	"""
	try:
		steps = ACTION_STEPS[action_name]
	except KeyError as exc:
		raise ValueError(f"Unknown action '{action_name}'") from exc
	print(f"Executing action '{action_name}'")
	_execute_steps(bot, steps)  # type: ignore[arg-type]
	time.sleep(1.0)


def main(argv: Iterable[str]) -> int:
	"""Parse CLI arguments and trigger the requested maneuver(s)."""
	parser = argparse.ArgumentParser(description="Test Raspbot turn sequences")
	parser.add_argument(
		"action",
		choices=sorted(ACTION_STEPS.keys()) + ["all"],
		help="Which maneuver to execute",
	)
	args = parser.parse_args(argv)

	# Add Car_main to sys.path for absolute import if needed
	import os
	import sys as _sys
	script_dir = os.path.dirname(os.path.abspath(__file__))
	car_main_dir = os.path.abspath(os.path.join(script_dir, ".."))
	if car_main_dir not in _sys.path:
		_sys.path.insert(0, car_main_dir)

	try:
		from devices.raspbot import Raspbot
	except Exception:
		print("Raspbot module unavailable; cannot execute maneuvers from CLI")
		return 1

	bot = Raspbot()
	if args.action == "all":
		for name in ACTION_STEPS:
			run_action(name, bot)
			time.sleep(2.0)
	else:
		run_action(args.action, bot)

	# Ensure motors are stopped at the end of the session
	_stop(bot)
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))

