"""Auto driving controller using lane detection and onboard camera."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Set

import cv2
import numpy as np

from devices.camera_node import Camera
from perception.lane import Lane
from perception.aruco_detector import ArUcoDetector
from control import planned_maneuvers

try:
	from devices.raspbot import Raspbot
except Exception:  # pragma: no cover - hardware optional during dev
	Raspbot = None  # type: ignore


LOG = logging.getLogger(__name__)


def _clamp(value: float, min_value: float, max_value: float) -> float:
	return max(min_value, min(max_value, value))


@dataclass
class LaneMeasurement:
	offset_cm: Optional[float]
	offset_pixels: Optional[float]
	overlay: Optional[np.ndarray]
	raw_frame: Optional[np.ndarray] = None


class DifferentialDrive:
	"""Wrapper around the Raspbot motor API with simple safeguards."""

	def __init__(self) -> None:
		self._bot = None
		if Raspbot is not None:
			try:
				self._bot = Raspbot()
				LOG.info("Initialized Raspbot motor controller")
			except Exception:
				LOG.exception("Failed to initialize Raspbot; motor commands disabled")
		else:
			LOG.warning("Raspbot module unavailable; motor commands disabled")

	def set_speeds(self, left: float, right: float) -> None:
		if self._bot is None:
			return
		for mid, speed in ((0, left), (1, left), (2, right), (3, right)):
			self._set_motor(mid, speed)

	def stop(self) -> None:
		if self._bot is None:
			return
		for mid in range(4):
			self._safe_ctrl(mid, 0, 0)

	def _set_motor(self, motor_id: int, signed_speed: float) -> None:
		direction = 0 if signed_speed >= 0 else 1
		magnitude = int(_clamp(abs(signed_speed), 0, 255))
		self._safe_ctrl(motor_id, direction, magnitude)

	def _safe_ctrl(self, motor_id: int, direction: int, speed: int) -> None:
		try:
			self._bot.Ctrl_Car(motor_id, direction, speed)  # type: ignore[union-attr]
		except Exception:
			LOG.exception("Ctrl_Car failed for motor %s", motor_id)

	def get_controller(self) -> Optional[Any]:
		return self._bot


class AutoModeController:
	"""Coordinate camera, lane detection, and drive control."""

	ARUCO_ACTION_HINTS: Dict[int, str] = {
		# Map marker IDs to semantic hints for intersection or lane behavior
		0: "start",
		1: "goal",
		2: "left_type_a",
		3: "left_type_b",
		4: "left_type_c",
		5: "right",
		6: "lane_change_left",
		7: "lane_change_right",
		8: "forward",
		9: "end_of_turn",
		10: "intersection",
		11: "stop",
	}

	def __init__(
		self,
		camera_index: int = 0,
		resolution: Tuple[int, int] = (640, 480),
		fps: Optional[int] = None,
		base_speed: int = 30,
		steering_gain: float = 0.45,
		control_rate_hz: float = 10.0,
		show_debug: bool = False,
		fallback_speed_ratio: float = 0.4,
		max_speed: int = 60,
	) -> None:
		width, height = resolution
		camera_args = dict(index=camera_index, width=width, height=height)
		if fps is not None:
			camera_args["fps"] = fps
		self.camera = Camera(**camera_args)
		self.drive = DifferentialDrive()
		self.base_speed = base_speed
		self.steering_gain = steering_gain
		self.control_interval = 1.0 / max(1.0, control_rate_hz)
		self.show_debug = show_debug
		self.fallback_speed_ratio = max(0.0, min(1.0, fallback_speed_ratio))
		self.max_speed = max(0, max_speed)
		self.planned_route: Optional[Dict[str, Any]] = None  # Injected from CLI planner
		self._raspbot = self.drive.get_controller()
		self.last_line_state: Optional[Tuple[int, int, int, int]] = None
		self._next_action_index = 0
		self._pending_action: Optional[str] = None
		self._pending_intersection_idx: Optional[int] = None
		self._aruco = ArUcoDetector(show_debug=show_debug)
		self._last_marker_ids: set[int] = set()
		self._marker_confidence_threshold = 0.5
		self._is_at_intersection = False
		self._intersection_action_pending = False
		self._pending_marker_hint: Optional[str] = None

	def run(self) -> None:
		if not self.camera.open():
			raise RuntimeError("Failed to open camera for auto mode")
		self.camera.start()
		LOG.info("Auto mode started")

		try:
			# Core control loop: read sensors, maintain lane, react to intersections and markers
			while True:
				ret, frame = self.camera.read(wait=True)
				if not ret or frame is None:
					time.sleep(0.01)
					continue

				line_state = self._read_line_sensors()
				is_all_white = bool(line_state == (1, 1, 1, 1)) if line_state is not None else False
				self._is_at_intersection = is_all_white
				if line_state is not None and line_state != self.last_line_state:
					self.last_line_state = line_state
					LOG.debug("IR line sensors: %s", line_state)
					if is_all_white:
						# All sensors see white: intersection detected, so pull the next planner action
						LOG.info("IR sensors detected intersection (all sensors white)")
						pending = self._next_planned_action()
						if pending:
							LOG.info("Upcoming planner action: %s", pending)
						# Immediately stop the vehicle at the white block (intersection)
						self.drive.stop()
						LOG.info("Stopped at intersection for 5.0s to await marker/hint")
						time.sleep(5.0)
						self._intersection_action_pending = True
					else:
						self._intersection_action_pending = False

				measurement = self._measure_lane(frame)
				self._apply_control(measurement)

				self._log_marker_detections(frame)
				if self._intersection_action_pending:
					self._handle_intersection()

				if self.show_debug:
					self._show_debug_frame(measurement)

				time.sleep(self.control_interval)
		except KeyboardInterrupt:
			LOG.info("Auto mode interrupted by user")
		finally:
			self.drive.stop()
			self.camera.close()
			if self.show_debug:
				cv2.destroyAllWindows()
			LOG.info("Auto mode stopped")

	def _measure_lane(self, frame: np.ndarray) -> LaneMeasurement:
		try:
			lane_obj = Lane(orig_frame=frame)
			lane_obj.get_line_markings()
			lane_obj.perspective_transform(plot=False)
			lane_obj.calculate_histogram(plot=False)
			left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
			if left_fit is None or right_fit is None:
				LOG.debug("Insufficient lane pixels; skipping frame")
				return LaneMeasurement(offset_cm=None, offset_pixels=None, overlay=None, raw_frame=frame)
			lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
			if lane_obj.left_fit is None or lane_obj.right_fit is None:
				LOG.debug("Refined lane fit failed; skipping frame")
				return LaneMeasurement(offset_cm=None, offset_pixels=None, overlay=None, raw_frame=frame)
			lane_obj.calculate_curvature()
			offset_cm = lane_obj.calculate_car_position()

			if lane_obj.left_fitx is None or lane_obj.right_fitx is None:
				return LaneMeasurement(offset_cm=None, offset_pixels=None, overlay=None, raw_frame=frame)

			lane_center = (lane_obj.left_fitx[-1] + lane_obj.right_fitx[-1]) / 2.0
			car_center = lane_obj.width / 2.0
			offset_pixels = car_center - lane_center
			overlay = lane_obj.overlay_lane_lines(plot=False)
			return LaneMeasurement(offset_cm=offset_cm, offset_pixels=offset_pixels, overlay=overlay, raw_frame=frame)
		except Exception:
			LOG.exception("Lane measurement failed")
			return LaneMeasurement(offset_cm=None, offset_pixels=None, overlay=None, raw_frame=frame)

	def _apply_control(self, measurement: LaneMeasurement) -> None:
		if measurement.offset_pixels is None:
			slow_speed = _clamp(self.base_speed * self.fallback_speed_ratio, -self.max_speed, self.max_speed)
			if slow_speed == 0:
				self.drive.stop()
			else:
				# IR-based fallback: nudge left/right if outer sensors see white
				left_nudge = right_nudge = slow_speed
				line_state = self._read_line_sensors()
				CORRECTION = 30  # Larger correction value
				CORRECTION_TIME = 0.25  # Correction duration in seconds (increase for longer nudge)
				COMPENSATE_RATIO = 0.7  # Compensation duration as a fraction of correction
				if line_state is not None:
					leftmost, _, _, rightmost = line_state
					if rightmost:
						# Nudge left
						left_nudge = max(0, slow_speed - CORRECTION)
						right_nudge = min(self.max_speed, slow_speed + CORRECTION)
						LOG.debug("IR fallback: rightmost sensor white, nudging left")
						self.drive.set_speeds(left_nudge, right_nudge)
						time.sleep(CORRECTION_TIME)
						self.drive.stop()
						time.sleep(0.05)
						# Compensate by nudging right
						comp_left = min(self.max_speed, slow_speed + CORRECTION)
						comp_right = max(0, slow_speed - CORRECTION)
						self.drive.set_speeds(comp_left, comp_right)
						time.sleep(CORRECTION_TIME * COMPENSATE_RATIO)
						self.drive.stop()
					elif leftmost:
						# Nudge right
						left_nudge = min(self.max_speed, slow_speed + CORRECTION)
						right_nudge = max(0, slow_speed - CORRECTION)
						LOG.debug("IR fallback: leftmost sensor white, nudging right")
						self.drive.set_speeds(left_nudge, right_nudge)
						time.sleep(CORRECTION_TIME)
						self.drive.stop()
						time.sleep(0.05)
						# Compensate by nudging left
						comp_left = max(0, slow_speed - CORRECTION)
						comp_right = min(self.max_speed, slow_speed + CORRECTION)
						self.drive.set_speeds(comp_left, comp_right)
						time.sleep(CORRECTION_TIME * COMPENSATE_RATIO)
						self.drive.stop()
					else:
						self.drive.set_speeds(left_nudge, right_nudge)
				else:
					self.drive.set_speeds(left_nudge, right_nudge)
				LOG.debug("No lane measurement; fallback speeds: L=%.1f R=%.1f", left_nudge, right_nudge)
			return

		turn = measurement.offset_pixels * self.steering_gain
		left_speed = self.base_speed - turn
		right_speed = self.base_speed + turn
		left_speed = _clamp(left_speed, -self.max_speed, self.max_speed)
		right_speed = _clamp(right_speed, -self.max_speed, self.max_speed)
		self.drive.set_speeds(left_speed, right_speed)

	def _show_debug_frame(self, measurement: LaneMeasurement) -> None:
		view_source = measurement.overlay if measurement.overlay is not None else measurement.raw_frame
		if view_source is None:
			return
		view = view_source.copy()
		if measurement.overlay is None:
			cv2.putText(view, "Lane detection unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		if measurement.offset_cm is not None:
			text = f"Offset: {measurement.offset_cm:.1f} cm"
			cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		cv2.imshow("Auto Mode - Lane", view)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			raise KeyboardInterrupt()

	def _read_line_sensors(self) -> Optional[Tuple[int, int, int, int]]:
		if self._raspbot is None:
			return None
		try:
			data = self._raspbot.read_data_array(0x0A, 1)
		except Exception:
			LOG.exception("Failed to read IR line sensors")
			return None
		if not data:
			return None
		try:
			track = int(data[0])
		except (TypeError, ValueError):
			LOG.warning("Unexpected IR sensor payload: %s", data)
			return None
		return (
			(track >> 3) & 0x01,
			(track >> 2) & 0x01,
			(track >> 1) & 0x01,
			track & 0x01,
		)

	def _log_marker_detections(self, frame: Optional[np.ndarray]) -> None:
		if frame is None:
			return
		markers = self._aruco.detect(frame)
		if not markers:
			if self._last_marker_ids:
				self._last_marker_ids.clear()
				LOG.debug("ArUco markers no longer visible")
			return
		high_conf_markers = [m for m in markers if m.confidence >= self._marker_confidence_threshold]
		for marker in markers:
			if marker.confidence < self._marker_confidence_threshold:
				LOG.debug(
					"Ignoring low-confidence marker id=%s confidence=%.3f",
					marker.marker_id,
					marker.confidence,
				)
		if not high_conf_markers:
			if self._last_marker_ids:
				self._last_marker_ids.clear()
				LOG.debug("All high-confidence markers lost")
			return
		current_ids = {marker.marker_id for marker in high_conf_markers}
		if current_ids != self._last_marker_ids:
			self._last_marker_ids = current_ids
			for marker in high_conf_markers:
				context = " [away from intersection]" if not self._is_at_intersection else ""
				LOG.info(
					"Detected ArUco marker id=%s center=%s confidence=%.3f%s",
					marker.marker_id,
					marker.center,
					marker.confidence,
					context,
				)
				hint = self.ARUCO_ACTION_HINTS.get(marker.marker_id)
				if hint:
					if self._is_at_intersection:
						LOG.debug("Marker hint %s recorded for intersection handling", hint)
						self._pending_marker_hint = hint
					else:
						LOG.info(
							"Marker id=%s confirmed outside intersection; hint=%s",
							marker.marker_id,
							hint,
						)
						self._handle_marker_hint_outside_intersection(hint)

	def _next_planned_action(self) -> Optional[str]:
		if self.planned_route is None:
			return None
		actions = self.planned_route.get("actions")
		indices = self.planned_route.get("intersections")
		if not actions or not indices:
			return None
		if self._next_action_index >= len(actions):
			LOG.info("Planner actions exhausted")
			return None
		action = actions[self._next_action_index]
		self._pending_action = action
		self._pending_intersection_idx = indices[self._next_action_index]
		self._next_action_index += 1
		LOG.debug(
			"Queued planner action %s at intersection index %s",
			action,
			self._pending_intersection_idx,
		)
		return action

	def _handle_intersection(self) -> None:
		if not self._intersection_action_pending:
			return
		if self._pending_action is None:
			LOG.info("Intersection detected but planner queue provided no action")
			self._clear_pending_intersection()
			return
		marker_hint = self._interpret_marker_hint(self._last_marker_ids)
		specific_hint = self._pending_marker_hint or marker_hint
		if specific_hint:
			LOG.info("ArUco marker hint at intersection: %s", marker_hint)
			if specific_hint.startswith("left_type_"):
				if self._pending_action == "left":
					LOG.info("Marker specifies %s, aligning with planner left turn", specific_hint)
				else:
					LOG.warning(
						"Marker requests %s but planner expects %s",
						specific_hint,
						self._pending_action,
					)
			elif specific_hint in {"left", "right", "forward"}:
				if marker_hint == self._pending_action:
					LOG.info("Marker hint matches planner action '%s'", self._pending_action)
				else:
					LOG.warning(
						"Marker hint %s differs from planner action %s",
						specific_hint,
						self._pending_action,
					)
			elif specific_hint in {"lane_change_left", "lane_change_right"}:
				LOG.info("Marker requests %s around the intersection", specific_hint.replace("_", " "))
			elif specific_hint == "stop":
				LOG.info("Marker indicates stop; planner action remains %s", self._pending_action)
			else:
				LOG.debug("Marker hint %s recorded for telemetry", specific_hint)
		else:
			LOG.info("No ArUco marker hint available at this intersection")
		LOG.info("Executing planner action '%s' (placeholder)", self._pending_action)
		if self._pending_action == "left":
			variant = specific_hint if specific_hint and specific_hint.startswith("left_type_") else None
			if variant:
				# execute the detected left-turn variant
				if self._raspbot is not None:
					planned_maneuvers.run_action(variant, self._raspbot)
				else:
					LOG.warning("No Raspbot controller available to run %s", variant)
			else:
				# execute a default left-turn variant when planner requests a generic left
				default_left = "left_type_b"
				if self._raspbot is not None:
					planned_maneuvers.run_action(default_left, self._raspbot)
				else:
					LOG.warning("No Raspbot controller available to run %s", default_left)
		elif self._pending_action == "right":
			variant = specific_hint if specific_hint == "right" else None
			if variant:
				if self._raspbot is not None:
					planned_maneuvers.run_action("right", self._raspbot)
				else:
					LOG.warning("No Raspbot controller available to run right turn")
			else:
				if self._raspbot is not None:
					planned_maneuvers.run_action("right", self._raspbot)
				else:
					LOG.warning("No Raspbot controller available to run right turn")
		elif self._pending_action == "forward":
			if self._raspbot is not None:
				planned_maneuvers.run_action("forward", self._raspbot)
			else:
				LOG.warning("No Raspbot controller available to run forward action")
		else:
			LOG.warning("Unknown planner action '%s'", self._pending_action)
 
		self._clear_pending_intersection()

	def _handle_marker_hint_outside_intersection(self, hint: str) -> None:
		if hint == "lane_change_left":
			if self._raspbot is not None:
				planned_maneuvers.run_action("lane_change_left", self._raspbot)
			else:
				LOG.warning("No Raspbot controller available to run lane_change_left")
		elif hint == "lane_change_right":
			if self._raspbot is not None:
				planned_maneuvers.run_action("lane_change_right", self._raspbot)
			else:
				LOG.warning("No Raspbot controller available to run lane_change_right")
		elif hint == "stop":
			# immediate controlled stop
			self.drive.stop()
			LOG.info("Performed immediate stop per marker request")
		elif hint in {"start", "goal", "end_of_turn", "intersection"}:
			LOG.debug("Context marker %s noted; no direct action required now", hint)
		elif hint.startswith("left_type_"):
			# TODO: prime the confirmed left-turn variant before reaching intersection
			LOG.debug("TODO: pre-stage parameters for %s", hint)
		elif hint == "right":
			# TODO: prime right-turn routine outside intersection if needed
			LOG.debug("TODO: pre-stage parameters for right turn")
		else:
			LOG.debug("Unhandled marker hint %s", hint)
	
	def _interpret_marker_hint(self, marker_ids: Set[int]) -> Optional[str]:
		if not marker_ids:
			return None
		hints = []
		for marker_id in marker_ids:
			hint = self.ARUCO_ACTION_HINTS.get(marker_id)
			if hint:
				hints.append(hint)
		if not hints:
			return None
		for candidate in (
			"left_type_a",
			"left_type_b",
			"left_type_c",
			"left",
			"right",
			"forward",
			"lane_change_left",
			"lane_change_right",
			"stop",
		):
			if candidate in hints:
				return candidate
		return hints[0]

	def _clear_pending_intersection(self) -> None:
		self._pending_action = None
		self._pending_intersection_idx = None
		self._intersection_action_pending = False
		self._pending_marker_hint = None
