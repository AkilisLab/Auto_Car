"""Manual test harness for validating Raspbot turn and lane-change routines."""


import argparse
import sys
import time
from typing import Iterable, Tuple, Any, Optional, List
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
SPIN_SPEED = 14  # further lowered for smoother, stable spins (was 18)
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



def forward_lanekeep(bot: Any, camera: Camera = None, duration: float = SHORT_FORWARD_TIME * 2, base_speed: float = FORWARD_SPEED, steering_gain: float = 0.45, max_speed: int = 255, show_debug: bool = False, stop_on_loss: bool = True) -> None:
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
				# Lane detection failed
				if stop_on_loss:
					# perform conspicuous early stop (fixed 3s) when caller requests it
					stop_dur = 3.0
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
				else:
					# caller requested to continue (e.g. we're about to spin); silently return
					return
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


def spin_until_lane(bot: Any, direction: str = "left", camera: Camera = None, timeout: float = 2.0, spin_speed: int = SPIN_SPEED, show_debug: bool = False, init_duration: float = 0.15, init_speed: int | None = None, debounce_frames: int = 2, fine_speed_factor: float = 0.4, detect_delay: float = 1.0) -> None:
	"""
	Spin the vehicle in-place until both lane lines are detected or `timeout` seconds elapse.
	`direction` should be either "left" or "right". If `camera` is None the function will
	fall back to a timed pivot using the existing step helpers.

	This routine is intended for manual testing and debugging via the CLI harness; it
	prints conspicuous logs and optionally shows overlays when `show_debug` is True.
	"""
	if bot is None:
		raise RuntimeError("Raspbot hardware unavailable; aborting spin_until_lane")
	# If no camera provided, fall back to a simple timed spin to preserve backwards compatibility
	if camera is None:
		print(f"No camera provided — performing timed spin {direction} for {timeout:.1f}s")
		if direction == "left":
			_execute_steps(bot, [_spin_left(timeout)])
		else:
			_execute_steps(bot, [_spin_right(timeout)])
		return

	print(f"Starting spin_until_lane: direction={direction} timeout={timeout:.1f}s speed={spin_speed}")
	# initial burst parameters
	if init_speed is None:
		init_speed = max(40, int(spin_speed * 1.8))
	# fine search speed derived from main spin_speed
	fine_speed = max(6, int(spin_speed * fine_speed_factor))
	# Initial burst commented out for stability
	# if init_duration and init_duration > 0:
	# 	print(f"Initial burst: dir={direction} speed={init_speed} dur={init_duration:.2f}s")
	# 	if direction == "left":
	# 		b_ldir, b_rdir = 1, 0
	# 	else:
	# 		b_ldir, b_rdir = 0, 1
	# 	for motor_id in (0, 1):
	# 		bot.Ctrl_Car(motor_id, b_ldir, init_speed)
	# 	for motor_id in (2, 3):
	# 		bot.Ctrl_Car(motor_id, b_rdir, init_speed)
	# 	time.sleep(init_duration)
	# 	_stop(bot)

	# Blind spin and detection delay commented out for stability
	# blind_spin_time = 0.18  # seconds (tunable)
	# print(f"Blind spin: {blind_spin_time:.2f}s with no lane detection")
	# if direction == "left":
	# 	ldir_blind, rdir_blind = 1, 0
	# else:
	# 	ldir_blind, rdir_blind = 0, 1
	# t_blind_end = time.time() + blind_spin_time
	# while time.time() < t_blind_end:
	# 	for motor_id in (0, 1):
	# 		bot.Ctrl_Car(motor_id, ldir_blind, spin_speed)
	# 	for motor_id in (2, 3):
	# 		bot.Ctrl_Car(motor_id, rdir_blind, spin_speed)
	# 	time.sleep(0.01)
	# _stop(bot)

	end_time = time.time() + timeout
	# motor directions for a pivot-in-place
	if direction == "left":
		ldir, rdir = 1, 0
	else:
		ldir, rdir = 0, 1

	consecutive = 0
	# start time for the spinning phase; we delay enabling lane detection until
	# `detect_delay` seconds have elapsed. This bypasses spurious detections
	# immediately after the spin starts.
	spin_start = time.time()
	try:
		while time.time() < end_time:
			# apply coarse spin command
			for motor_id in (0, 1):
				bot.Ctrl_Car(motor_id, ldir, spin_speed)
			for motor_id in (2, 3):
				bot.Ctrl_Car(motor_id, rdir, spin_speed)

			# If we're still within the initial detection delay window, skip detection
			if time.time() - spin_start < detect_delay:
				# continue spinning without sampling the detector; small sleep to yield
				time.sleep(0.01)
				continue
			# read a frame and attempt detection
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
			except Exception:
				# detection error, keep spinning
				consecutive = 0
				time.sleep(0.01)
				continue

			# If both fits present, increment the debounce counter
			if left_fit is not None and right_fit is not None:
				consecutive += 1
			else:
				consecutive = 0

			if consecutive >= debounce_frames:
				# Stable detection — refine and compute offset
				print(f"Lanes detected ({consecutive} frames) — stopping spin (direction={direction})")
				try:
					lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
				except Exception:
					pass
				offset_pixels = None
				try:
					if getattr(lane_obj, "left_fitx", None) is not None and getattr(lane_obj, "right_fitx", None) is not None:
						lane_center = (lane_obj.left_fitx[-1] + lane_obj.right_fitx[-1]) / 2.0
						car_center = lane_obj.width / 2.0
						offset_pixels = car_center - lane_center
				except Exception:
					offset_pixels = None

				# stop main spin
				_stop(bot)
				# debug overlay
				if show_debug:
					try:
						vis = lane_obj.overlay_lane_lines(plot=False) or frame
						cv2.putText(vis, "LANES FOUND - SPIN STOPPED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
						cv2.imshow("Maneuver - SpinUntilLane", vis)
						cv2.waitKey(200)
					except Exception:
						pass

				# If we have an offset, apply small opposite-direction compensation in steps
				if offset_pixels is not None:
					norm = min(1.0, abs(offset_pixels) / (lane_obj.width / 2.0)) if getattr(lane_obj, "width", None) else 0.0
					comp_speed = 15 # max(15, int(spin_speed * 0.45))
					comp_duration = 0.5 # max(0.3, min(0.6, norm * 0.6))
					step_dt = 0.05
					steps = max(1, int(comp_duration / step_dt))
					# opposite-direction
					if direction == "left":
						c_ldir, c_rdir = 0, 1
					else:
						c_ldir, c_rdir = 1, 0
					print(f"Applying incremental compensation spin: dir={'right' if direction=='left' else 'left'} speed={comp_speed} total_dur={comp_duration:.2f}s steps={steps} norm={norm:.2f}")
					# step loop
					comp_stopped_early = False
					for _ in range(steps):
						for motor_id in (0, 1):
							bot.Ctrl_Car(motor_id, c_ldir, comp_speed)
						for motor_id in (2, 3):
							bot.Ctrl_Car(motor_id, c_rdir, comp_speed)
						time.sleep(comp_duration / steps)
						# quick check to see whether lanes are now solidly visible
						ret2, f2 = camera.read(wait=True)
						if not (ret2 and f2 is not None):
							continue
						try:
							lobj2 = Lane(orig_frame=f2)
							lobj2.get_line_markings()
							lobj2.perspective_transform(plot=False)
							lobj2.calculate_histogram(plot=False)
							l2, r2 = lobj2.get_lane_line_indices_sliding_windows(plot=False)
							if l2 is not None and r2 is not None:
								print("Compensation: lanes detected early — attempting settle oscillation")
								_stop(bot)
								# Instead of bursting forward, perform small oscillatory micro-spins
								# until we observe `debounce_frames` consecutive good detections or exhaust attempts.
								settle_attempts = 12
								settle_step_dur = 0.1
								settle_speed = max(13, int(spin_speed * 0.35))
								stable_count = 0
								for attempt in range(settle_attempts):
									# alternate tiny spins: even attempts spin correction direction, odd attempts the other way
									if attempt % 2 == 0:
										for motor_id in (0, 1):
											bot.Ctrl_Car(motor_id, c_ldir, settle_speed)
										for motor_id in (2, 3):
											bot.Ctrl_Car(motor_id, c_rdir, settle_speed)
									else:
										for motor_id in (0, 1):
											bot.Ctrl_Car(motor_id, ldir, settle_speed)
										for motor_id in (2, 3):
											bot.Ctrl_Car(motor_id, rdir, settle_speed)
									time.sleep(settle_step_dur)
									_stop(bot)
									# check if lanes are stable now
									ret3, f3 = camera.read(wait=True)
									if not (ret3 and f3 is not None):
										stable_count = 0
										continue
									try:
										lobj3 = Lane(orig_frame=f3)
										lobj3.get_line_markings()
										lobj3.perspective_transform(plot=False)
										lobj3.calculate_histogram(plot=False)
										ll3, rr3 = lobj3.get_lane_line_indices_sliding_windows(plot=False)
										if ll3 is not None and rr3 is not None:
											stable_count += 1
											if stable_count >= debounce_frames:
												print("Settle oscillation: stable lanes confirmed")
												comp_stopped_early = True
												break
										else:
											stable_count = 0
									except Exception:
										stable_count = 0
								# end for attempts
								if not comp_stopped_early:
									# couldn't confirm stable lanes — ensure motors stopped and continue
									_stop(bot)
								else:
									# we confirmed stable lanes; leave motors stopped and return to caller
									_stop(bot)
								break
						except Exception:
							# ignore detection errors during compensation
							continue
					# end for steps
					if not comp_stopped_early:
						# Couldn't confirm stability during compensation - stop motors and try short settle oscillation
						_stop(bot)
						settle_attempts = 8
						settle_step_dur = 0.05
						settle_speed = max(6, int(spin_speed * 0.3))
						stable_count = 0
						for attempt in range(settle_attempts):
							# alternate tiny corrective spins
							if attempt % 2 == 0:
								for motor_id in (0, 1):
									bot.Ctrl_Car(motor_id, c_ldir, settle_speed)
								for motor_id in (2, 3):
									bot.Ctrl_Car(motor_id, c_rdir, settle_speed)
							else:
								for motor_id in (0, 1):
									bot.Ctrl_Car(motor_id, ldir, settle_speed)
								for motor_id in (2, 3):
									bot.Ctrl_Car(motor_id, rdir, settle_speed)
							time.sleep(settle_step_dur)
							_stop(bot)
							ret3, f3 = camera.read(wait=True)
							if not (ret3 and f3 is not None):
								stable_count = 0
								continue
							try:
								lobj3 = Lane(orig_frame=f3)
								lobj3.get_line_markings()
								lobj3.perspective_transform(plot=False)
								lobj3.calculate_histogram(plot=False)
								ll3, rr3 = lobj3.get_lane_line_indices_sliding_windows(plot=False)
								if ll3 is not None and rr3 is not None:
									stable_count += 1
									if stable_count >= debounce_frames:
										print("Final settle: stable lanes confirmed")
										break
								else:
									stable_count = 0
							except Exception:
								stable_count = 0
						# end settle attempts
						_stop(bot)
				else:
					# no offset information; allow a short settle pause
					time.sleep(0.08)
				return

			# small sleep to avoid tight loop
			time.sleep(0.01)

		# timeout reached: ensure stop and notify
		print(f"Spin timeout reached ({timeout:.1f}s) without detecting both lanes — stopping")
		_stop(bot)
		if show_debug:
			try:
				cv2.destroyWindow("Maneuver - SpinUntilLane")
			except Exception:
				pass
	except Exception:
		# ensure motors are stopped on unexpected errors
		_stop(bot)
		raise


def drive_until_marker(bot: Any, camera: Camera = None, marker_ids: Optional[List[int]] = None, forward_speed: int = FORWARD_SPEED, check_interval: float = 0.05, timeout: Optional[float] = 6.0, show_debug: bool = False) -> None:
	"""
	Drive forward (open-loop) until an ArUco marker is detected inside the detector ROI.
	If `camera` is None, performs a timed forward run for `timeout` seconds (if provided).
	`marker_ids` can be a list of acceptable marker ids; if None any detected marker will stop the drive.
	"""
	if bot is None:
		raise RuntimeError("Raspbot hardware unavailable; aborting drive_until_marker")
	# If no camera, fallback to timed forward (so caller doesn't block forever)
	if camera is None:
		if timeout is None:
			# nothing to do without camera and no timeout
			return
		_execute_steps(bot, [(0, int(forward_speed * 1.0), 0, int(forward_speed * 1.0), float(timeout))])
		return
	# dynamic import to avoid top-level circular deps
	try:
		from perception.aruco_detector import ArUcoDetector
	except Exception:
		ArUcoDetector = None

	detector = ArUcoDetector(show_debug=show_debug) if ArUcoDetector is not None else None
	end_time = time.time() + timeout if timeout is not None else None
	try:
		while True:
			# drive forward
			for motor_id in (0, 1):
				bot.Ctrl_Car(motor_id, 0, int(forward_speed * 1.0))
			for motor_id in (2, 3):
				bot.Ctrl_Car(motor_id, 0, int(forward_speed * 1.0))
			# capture a frame and check for markers if detector available
			ret, frame = camera.read(wait=True)
			if ret and frame is not None and detector is not None:
				try:
					markers = detector.detect(frame)
					if markers:
						ids = [m.marker_id for m in markers]
						# require marker id 0 for safety
						if 0 in ids:
							_stop(bot)
							return
						# unexpected marker(s) detected — log and continue driving
						print(f"Unexpected ArUco marker(s) detected: {ids} — ignoring and continuing forward")
				except Exception:
					# ignore detection errors and continue driving
					pass

			# check for timeout
			if end_time is not None and time.time() >= end_time:
				_stop(bot)
				return
			time.sleep(check_interval)
	finally:
		# emergency stop
		_stop(bot)



ACTION_STEPS = {
	"left": (
		# ("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		("drive_until_marker", {"marker_ids": None, "timeout": 8.0}),
		(0, int(FORWARD_SPEED * 1.0), 0, int(FORWARD_SPEED * 1.0), 0.25),
		("spin_until_lane", {"direction": "left", "timeout": SPIN_TIME * 3, "spin_speed": SPIN_SPEED}),
	),
	"right": (
		# ("forward_lanekeep", {"duration": SHORT_FORWARD_TIME}),
		("drive_until_marker", {"marker_ids": None, "timeout": 8.0}),
		(0, int(FORWARD_SPEED * 1.0), 0, int(FORWARD_SPEED * 1.0), 0.25),
		("spin_until_lane", {"direction": "right", "timeout": SPIN_TIME * 3, "spin_speed": SPIN_SPEED}),
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
			# Dispatch callable-style maneuver steps
			if func_name == "forward_lanekeep":
				# ensure show_debug flag is passed through
				kwargs.setdefault("show_debug", show_debug)
				forward_lanekeep(bot, camera=camera, **kwargs)
			elif func_name == "spin_until_lane":
				# spin until both lanes visible or timeout
				kwargs.setdefault("show_debug", show_debug)
				# For left spins, disable detection delay (start lane detection immediately)
				if kwargs.get("direction") == "left":
					kwargs["detect_delay"] = 0.0
				# For right spins, use a larger detection delay
				elif kwargs.get("direction") == "right":
					kwargs["detect_delay"] = 1.2
				spin_until_lane(bot, camera=camera, **kwargs)
			elif func_name == "drive_until_marker":
				# open-loop forward until an ArUco marker appears in the detector ROI
				kwargs.setdefault("show_debug", show_debug)
				drive_until_marker(bot, camera=camera, **kwargs)
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

