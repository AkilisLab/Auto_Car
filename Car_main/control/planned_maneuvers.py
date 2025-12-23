"""Manual test harness for validating Raspbot turn and lane-change routines."""


import argparse
import sys
import time
from typing import Iterable, Tuple, Any
import cv2
import random

# Ensure absolute import paths for Camera and Lane
import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
_car_main_dir = _os.path.abspath(_os.path.join(_script_dir, ".."))
if _car_main_dir not in sys.path:
	sys.path.insert(0, _car_main_dir)

# Import Camera and Lane using absolute paths
from devices.camera_node import Camera
from perception.lane import Lane

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


def forward_lanekeep(bot: Any, camera: Camera = None, duration: float = SHORT_FORWARD_TIME * 2, base_speed: float = FORWARD_SPEED, steering_gain: float = 0.45, max_speed: int = 255, show_debug: bool = False) -> None:
	"""
	Drive forward while keeping lane using camera+lane detector for `duration` seconds.
	If `camera` is None, falls back to simple timed forward motion.
	"""
	if bot is None:
		raise RuntimeError("Raspbot hardware unavailable; aborting forward_lanekeep")
	end_time = time.time() + duration
	while time.time() < end_time:
		if camera is None:
			# fallback: simple forward for a short interval
			for motor_id in (0, 1):
				bot.Ctrl_Car(motor_id, 0, int(base_speed * 1.1))
			for motor_id in (2, 3):
				bot.Ctrl_Car(motor_id, 0, int(base_speed))
			time.sleep(0.05)
			continue
		# Get frame and try lane detection
		ret, frame = camera.read(wait=True)
		if not ret or frame is None:
			time.sleep(0.01)
			continue
		try:
			lane_obj = Lane(orig_frame=frame)
			lane_obj.get_line_markings()
			lane_obj.perspective_transform(plot=False)
			lane_obj.calculate_histogram(plot=False)
			left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
			overlay = None
			if left_fit is None or right_fit is None:
				# Lane detection failed — perform conspicuous early stop (5-10s)
				stop_dur = random.uniform(5.0, 10.0)
				ts = time.strftime("%Y-%m-%d %H:%M:%S")
				print("\n" + "*" * 60)
				print(f"*** EARLY STOP: Lane lost at {ts} — stopping for {stop_dur:.1f}s ***")
				print("*" * 60 + "\n")
				_stop(bot)
				if show_debug:
					try:
						vis = frame.copy()
						cv2.putText(vis, f"LANE LOST - STOP {stop_dur:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
						cv2.imshow("Maneuver - Forward LaneKeep", vis)
						# ensure the overlay is drawn at least once
						cv2.waitKey(1)
					except Exception:
						pass
				# pause for a conspicuous duration so the stop is noticeable/safe
				time.sleep(stop_dur)
				break
			lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
			if lane_obj.left_fitx is None or lane_obj.right_fitx is None:
				for motor_id in (0, 1):
					bot.Ctrl_Car(motor_id, 0, int(base_speed * 1.1))
				for motor_id in (2, 3):
					bot.Ctrl_Car(motor_id, 0, int(base_speed))
				time.sleep(0.01)
				continue
			# compute offset and steering correction (same logic as AutoModeController._apply_control)
			lane_center = (lane_obj.left_fitx[-1] + lane_obj.right_fitx[-1]) / 2.0
			car_center = lane_obj.width / 2.0
			offset_pixels = car_center - lane_center
			turn = offset_pixels * steering_gain
			left_speed = base_speed - turn
			right_speed = base_speed + turn
			# clamp and send to motors
			lmag = int(max(0, min(max_speed, abs(left_speed))))
			rmag = int(max(0, min(max_speed, abs(right_speed))))
			ldir = 0 if left_speed >= 0 else 1
			rdir = 0 if right_speed >= 0 else 1
			for motor_id in (0, 1):
				bot.Ctrl_Car(motor_id, ldir, lmag)
			for motor_id in (2, 3):
				bot.Ctrl_Car(motor_id, rdir, rmag)
			# optionally create overlay and show
			if show_debug:
				try:
					overlay = lane_obj.overlay_lane_lines(plot=False)
				except Exception:
					overlay = None
				view = overlay if overlay is not None else frame
				cv2.imshow("Maneuver - Forward LaneKeep", view)
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		except Exception:
			# on detection error, drive forward open-loop briefly
			for motor_id in (0, 1):
				bot.Ctrl_Car(motor_id, 0, int(base_speed * 1.1))
			for motor_id in (2, 3):
				bot.Ctrl_Car(motor_id, 0, int(base_speed))
			time.sleep(0.01)
		# small wait to avoid busy loop
		time.sleep(0.01)

	# ensure stop at the end
	_stop(bot)
	time.sleep(SHORT_GAP)


ACTION_STEPS = {
	"left_type_a": (
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME * 1.7}),
		_spin_left(SPIN_TIME),
	),
	"left_type_b": (
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME * 1.7}),
		_spin_left(SPIN_TIME),
	),
	"left_type_c": (
		("forward_lanekeep", {"duration": LONG_FORWARD_TIME}),
		_spin_left(SPIN_TIME),
		# ("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
	),
	"right": (
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		_spin_right(SPIN_TIME),
	),
	# Use lane-keeping forward action (will use camera when provided to run_action)
	"forward": (
		("forward_lanekeep", {"duration": LONG_FORWARD_TIME}),
	),
	"lane_change_left": (
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		_arc_left(ARC_TIME),
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		_arc_right(ARC_TIME),
	),
	"lane_change_right": (
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		_arc_right(ARC_TIME),
		("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		_arc_left(ARC_TIME),
	),
}


def run_action(action_name: str, bot: Any, camera: Camera = None, show_debug: bool = False) -> None:
	"""Play back the scripted maneuver on the provided `bot` controller.

	The caller (for example `AutoModeController`) should pass the Raspbot
	controller instance (available via `drive.get_controller()`). This
	function will raise if `bot` is None.

	If `camera` is provided, certain actions (like `forward_lanekeep`) will
	use it for lane detection; otherwise they fall back to open-loop behavior.
	"""
	try:
		steps = ACTION_STEPS[action_name]
	except KeyError as exc:
		raise ValueError(f"Unknown action '{action_name}'") from exc
	print(f"Executing action '{action_name}'")
	for step in steps:
		# Callable-style step: ("name", kwargs)
		if isinstance(step, tuple) and len(step) == 2 and isinstance(step[0], str):
			func_name, kwargs = step
			if func_name == "forward_lanekeep":
				# ensure show_debug flag is passed through
				kwargs.setdefault("show_debug", show_debug)
				forward_lanekeep(bot, camera=camera, **kwargs)
			else:
				raise ValueError(f"Unknown maneuver function: {func_name}")
		# Regular Step tuple
		elif isinstance(step, tuple):
			_execute_steps(bot, [step])
		else:
			raise ValueError(f"Invalid step in ACTION_STEPS: {step}")

	# Close any OpenCV windows if we were showing debug
	try:
		if show_debug:
			cv2.destroyAllWindows()
	except Exception:
		pass
	time.sleep(1.0)


def main(argv: Iterable[str]) -> int:
	"""Parse CLI arguments and trigger the requested maneuver(s)."""
	parser = argparse.ArgumentParser(description="Test Raspbot turn sequences")
	parser.add_argument(
		"action",
		choices=sorted(ACTION_STEPS.keys()) + ["all"],
		help="Which maneuver to execute",
	)
	parser.add_argument("--show", action="store_true", help="Open camera and visualize lane detection during maneuvers")
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
	cam = None
	if args.show:
		try:
			cam = Camera(index=0, width=640, height=480)
			if not cam.open():
				print("Failed to open camera for visualization")
				cam = None
			else:
				cam.start()
		except Exception:
			cam = None
	if args.action == "all":
		for name in ACTION_STEPS:
			run_action(name, bot, camera=cam, show_debug=args.show)
			time.sleep(2.0)
	else:
		run_action(args.action, bot, camera=cam, show_debug=args.show)

	# Ensure motors are stopped at the end of the session
	_stop(bot)
	if cam is not None:
		cam.stop()
		cam.close()
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))

