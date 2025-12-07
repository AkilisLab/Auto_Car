#!/usr/bin/env python3
"""Auto lane-following split into three responsibilities:

- PerceptionInput: camera / file input and frame acquisition (provides
  RGB frames for algorithms and BGR frames for display/recording).
- LaneDetection: image processing and lane center detection.
- CarController: maps detected lane center -> motor commands, keeps
  memory and safety logic, and exposes autodrive toggle.

The file also provides a small `LaneFollower` coordinator that wires
these components together and a CLI entrypoint preserving the
previous command-line options.

This refactor keeps behavior unchanged while isolating responsibilities
so tests and future improvements (color-based detection, PID control,
different vehicle interface) are easier to add.
"""

import argparse
import time
import collections
import logging
from typing import Optional, Tuple

import cv2
import os
import numpy as np

from devices.raspbot import Raspbot
from perception.lane_detector import LaneDetection


def clamp(v, a, b):
    return max(a, min(b, v))


class PerceptionInput:
    """Provides frames from a camera index or video file.

    When a camera index (int) is used this class prefers the
    `devices.camera_node.Camera` background reader and exposes
    `read_rgb()` which returns RGB-ordered frames for algorithms
    while still allowing BGR display conversion for OpenCV windows.
    """

    def __init__(self, source=0):
        self.source = source
        self.cam = None
        self.cap = None
        self.input_is_rgb = False

    def start(self) -> None:
        # if source is integer try Camera wrapper
        if isinstance(self.source, int):
            try:
                from devices.camera_node import Camera

                self.cam = Camera(index=self.source)
                if not self.cam.start():
                    # fallback to cv2 VideoCapture
                    self.cam = None
                else:
                    self.input_is_rgb = True
                    return
            except Exception:
                self.cam = None

        # fallback to direct VideoCapture
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {self.source}")
        self.input_is_rgb = False

    def read(self, wait: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (ret, frame).

        When `input_is_rgb` True frames are RGB-ordered, otherwise BGR.
        """
        if self.cam is not None:
            return self.cam.read_rgb(wait=wait)
        if self.cap is None:
            # attempt to open lazily
            self.start()
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def stop(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        try:
            if self.cam is not None:
                self.cam.stop()
                self.cam.close()
                self.cam = None
        except Exception:
            pass

    def frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Return a BGR frame suitable for cv2.imshow/VideoWriter.

        If internal frames are RGB this converts back to BGR; otherwise
        returns a copy of the input.
        """
        if frame is None:
            return None
        if self.input_is_rgb:
            try:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                return frame.copy()
        return frame.copy()





class CarController:
    """Maps lane center information to motor commands.

    Responsibility: compute steering differential, apply global speed
    scaling, manage memory and autodrive toggle, and perform safe stop.
    """

    def __init__(self, bot: Optional[Raspbot] = None, base_speed: int = 80,
                 steering_gain: float = 0.9, speed_scale: float = 0.6,
                 lost_timeout: float = 1.0, memory_timeout: float = 3.0,
                 memory_decay_min: float = 0.3,
                 speed_sensitivity: float = 0.8, min_forward_scale: float = 0.3):
        self.bot = bot or Raspbot()
        self.base_speed = int(clamp(base_speed, 0, 255))
        self.steering_gain = float(steering_gain)
        self.speed_scale = float(clamp(speed_scale, 0.0, 1.0))
        self.last_center = None
        self.last_seen_time = time.time()
        self.lost_timeout = float(lost_timeout)
        self.memory = None
        self.memory_timeout = float(memory_timeout)
        self.memory_decay_min = float(memory_decay_min)
        # How aggressively to reduce forward speed for larger steering error.
        # 0.0 means no reduction, 1.0 means scale = 1 - |error| (full reduction).
        self.speed_sensitivity = float(clamp(speed_sensitivity, 0.0, 5.0))
        # Minimum forward scale (to avoid stopping completely while turning)
        self.min_forward_scale = float(clamp(min_forward_scale, 0.0, 1.0))
        self.autodrive = True

    def set_motors(self, left_speed: float, right_speed: float) -> None:
        ls = int(clamp(int(left_speed * self.speed_scale), 0, 255))
        rs = int(clamp(int(right_speed * self.speed_scale), 0, 255))
        try:
            self.bot.Ctrl_Car(0, 0, ls)
            self.bot.Ctrl_Car(1, 0, ls)
            self.bot.Ctrl_Car(2, 0, rs)
            self.bot.Ctrl_Car(3, 0, rs)
        except Exception:
            logging.exception("Error sending motor commands")

    def stop(self) -> None:
        try:
            self.set_motors(0, 0)
        finally:
            pass

    def update_from_detection(self, lane_center: int, frame_w: int) -> None:
        """Compute control from a visible lane center and update memory."""
        frame_center = frame_w // 2
        error = (frame_center - lane_center) / float(frame_center)
        # reduce forward base according to absolute steering error so the
        # car slows down on sharper curves
        forward_scale = 1.0 - (self.speed_sensitivity * abs(error))
        forward_scale = max(self.min_forward_scale, min(1.0, forward_scale))
        adj_base = self.base_speed * forward_scale
        diff = self.steering_gain * error * adj_base
        left_speed = adj_base - diff
        right_speed = adj_base + diff
        # update memory
        self.memory = {'center': lane_center, 'ts': time.time(), 'error': error, 'diff': diff}
        self.last_center = lane_center
        self.last_seen_time = time.time()
        if self.autodrive:
            self.set_motors(left_speed, right_speed)

    def handle_missing(self, show_debug: bool, frame: Optional[np.ndarray] = None) -> bool:
        """Called when detection returns None.

        Uses memory guidance if available and recent. Returns True to
        continue running, or False to signal a stop (e.g., lost too long).
        """
        time_since_seen = time.time() - self.last_seen_time
        if self.memory is not None and time_since_seen <= self.memory_timeout:
            mem = self.memory
            decay = max(self.memory_decay_min, 1.0 - (time_since_seen / self.memory_timeout))
            # apply forward slowdown proportional to historical error
            mem_error = abs(mem.get('error', 0.0))
            forward_scale = 1.0 - (self.speed_sensitivity * mem_error)
            forward_scale = max(self.min_forward_scale, min(1.0, forward_scale))
            adj_base = self.base_speed * forward_scale
            mem_diff = mem.get('diff', 0.0) * decay
            left_speed = adj_base - mem_diff
            right_speed = adj_base + mem_diff
            if self.autodrive:
                self.set_motors(left_speed, right_speed)
            if show_debug and frame is not None:
                try:
                    cv2.putText(frame, 'Using memory guidance', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except Exception:
                    pass
            return True
        if time_since_seen > self.lost_timeout:
            logging.info("Lane lost for too long -> stopping")
            self.stop()
            return False
        # gentle fallback: don't update motors, caller can choose to keep last
        return True


class LaneFollower:
    """Coordinator: wires Perception, Detection and Control into a loop."""

    def __init__(self, video_source=0, base_speed=80, steering_gain=0.9, speed_scale=0.6, speed_sensitivity: float = 0.8, min_forward: float = 0.3, view: str = 'center', calib_path: str = None, route=None, turning_points=None, intersection_indices=None, intersection_actions=None):
        self.perception = PerceptionInput(video_source)
        self.detector = LaneDetection()
        try:
            self.detector.set_view(view)
        except Exception:
            pass
        # load calibration if provided (expects .npz with camera_matrix and dist_coefs)
        if calib_path is not None:
            try:
                if os.path.isfile(calib_path):
                    npz = np.load(calib_path)
                    cam = npz.get('camera_matrix')
                    dist = npz.get('dist_coefs')
                    if cam is not None and dist is not None:
                        self.detector.set_camera_intrinsics(cam, dist)
            except Exception:
                logging.exception('Failed to load calibration file: %s', calib_path)

        self.controller = CarController(base_speed=base_speed, steering_gain=steering_gain, speed_scale=speed_scale, speed_sensitivity=speed_sensitivity, min_forward_scale=min_forward)
        # Store route and intersection logic
        self.route = route if route is not None else []
        self.intersection_indices = intersection_indices if intersection_indices is not None else []
        self.intersection_actions = intersection_actions if intersection_actions is not None else []
        self.current_intersection = 0  # index into intersection_indices/actions
        self.last_ir_state = None
        self.ir_triggered = False

    def run(self, show_debug: bool = False, show_steps: bool = False, save_steps: bool = False):
        self.perception.start()
        frame_idx = 0
        try:
            while True:
                # --- IR sensor check for intersection ---
                try:
                    ir_data = self.controller.bot.read_data_array(0x0a, 1)
                    ir_val = int(ir_data[0]) if ir_data and len(ir_data) > 0 else 0
                except Exception:
                    ir_val = 0
                # Trigger only if all four IR sensors are active (all bits set)
                at_intersection = (ir_val & 0x0F) == 0x0F
                # Rising edge detection
                if at_intersection and not self.ir_triggered:
                    self.ir_triggered = True
                    # Handle intersection event
                    if self.current_intersection < len(self.intersection_indices):
                        action = self.intersection_actions[self.current_intersection] if self.current_intersection < len(self.intersection_actions) else 'forward'
                        # Map action to view
                        if action == 'left':
                            view = 'left'
                        elif action == 'right':
                            view = 'right'
                        else:
                            view = 'center'
                        try:
                            self.detector.set_view(view)
                        except Exception:
                            pass
                        # Set camera servo angle (assuming servo id 1)
                        angle = self.detector._view_rotations.get(view, 0.0)
                        # Map angle (e.g., -45, 0, 45) to servo range (0-180)
                        # Example: center=90, left=135, right=45
                        servo_angle = 90
                        if view == 'left':
                            servo_angle = 135
                        elif view == 'right':
                            servo_angle = 45
                        try:
                            self.controller.bot.Ctrl_Servo(1, int(servo_angle))
                        except Exception:
                            pass
                        logging.info(f"Intersection {self.current_intersection}: action={action}, view={view}, servo={servo_angle}")
                        self.current_intersection += 1
                elif not at_intersection:
                    self.ir_triggered = False

                ret, frame = self.perception.read(wait=True)
                if not ret:
                    logging.info('End of stream or cannot read frame')
                    break

                # Optionally run the detector with debug outputs (intermediate images)
                debug = None
                if show_steps or show_debug:
                    lane_center, debug = self.detector.detect_with_debug(frame, is_rgb=self.perception.input_is_rgb)
                else:
                    lane_center = self.detector.detect_lane_center(frame, is_rgb=self.perception.input_is_rgb)

                if lane_center is None:
                    # if debug images are present, optionally show/save them
                    if debug is not None:
                        try:
                            if 'edges' in debug and debug['edges'] is not None and show_steps:
                                cv2.imshow('edges', debug['edges'])
                            if 'roi' in debug and debug['roi'] is not None and show_steps:
                                cv2.imshow('roi', debug['roi'])
                            if 'hough' in debug and debug['hough'] is not None and show_debug:
                                cv2.imshow('hough', debug['hough'])
                            if save_steps:
                                os.makedirs('debug_out', exist_ok=True)
                                ts = int(time.time() * 1000)
                                if 'proc' in debug and debug['proc'] is not None:
                                    cv2.imwrite(f'debug_out/proc_{ts}_{frame_idx}.png', debug['proc'])
                                if 'edges' in debug and debug['edges'] is not None:
                                    cv2.imwrite(f'debug_out/edges_{ts}_{frame_idx}.png', debug['edges'])
                                if 'roi' in debug and debug['roi'] is not None:
                                    cv2.imwrite(f'debug_out/roi_{ts}_{frame_idx}.png', debug['roi'])
                                if 'hough' in debug and debug['hough'] is not None:
                                    cv2.imwrite(f'debug_out/hough_{ts}_{frame_idx}.png', debug['hough'])
                        except Exception:
                            pass

                    cont = self.controller.handle_missing(show_debug=show_debug, frame=(self.perception.frame_for_display(frame) if show_debug else None))
                    if not cont:
                        break
                    # continue to next frame
                    if show_debug:
                        vis = self.perception.frame_for_display(frame)
                        cv2.imshow('lane', vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('p'):
                            break
                        if key == ord('s'):
                            self.controller.autodrive = not self.controller.autodrive
                            if not self.controller.autodrive:
                                self.controller.stop()
                            logging.info('Autodrive toggled: %s', 'ON' if self.controller.autodrive else 'OFF')
                    continue

                # visible lane: update controller and optionally show overlay
                self.controller.update_from_detection(lane_center, frame.shape[1])
                # if we have debug images, show/save them
                if debug is not None:
                    try:
                        if 'hough' in debug and debug['hough'] is not None and show_debug:
                            cv2.imshow('hough', debug['hough'])
                        if save_steps:
                            os.makedirs('debug_out', exist_ok=True)
                            ts = int(time.time() * 1000)
                            if 'proc' in debug and debug['proc'] is not None:
                                cv2.imwrite(f'debug_out/proc_{ts}_{frame_idx}.png', debug['proc'])
                            if 'edges' in debug and debug['edges'] is not None:
                                cv2.imwrite(f'debug_out/edges_{ts}_{frame_idx}.png', debug['edges'])
                            if 'roi' in debug and debug['roi'] is not None:
                                cv2.imwrite(f'debug_out/roi_{ts}_{frame_idx}.png', debug['roi'])
                            if 'hough' in debug and debug['hough'] is not None:
                                cv2.imwrite(f'debug_out/hough_{ts}_{frame_idx}.png', debug['hough'])
                    except Exception:
                        pass
                if show_debug:
                    vis = self.perception.frame_for_display(frame)
                    # draw overlay
                    cv2.line(vis, (vis.shape[1] // 2, vis.shape[0]), (lane_center, int(vis.shape[0] * 0.6)), (0, 255, 0), 2)
                    cv2.circle(vis, (lane_center, vis.shape[0] - 10), 6, (0, 0, 255), -1)
                    cv2.imshow('lane', vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        self.controller.autodrive = not self.controller.autodrive
                        if not self.controller.autodrive:
                            self.controller.stop()
                        logging.info('Autodrive toggled: %s', 'ON' if self.controller.autodrive else 'OFF')
                frame_idx += 1

        except KeyboardInterrupt:
            logging.info('Keyboard interrupt - stopping')
        finally:
            self.controller.stop()
            self.perception.stop()
            if show_debug:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass


def parse_args():
    p = argparse.ArgumentParser(description='Auto lane-following mode (refactored)')
    p.add_argument('--video', '-v', default=0, help='video source (camera index or file path)')
    p.add_argument('--speed', '-s', type=int, default=80, help='base motor speed (0..255)')
    p.add_argument('--gain', '-g', type=float, default=0.9, help='steering gain')
    p.add_argument('--scale', type=float, default=0.6, help='global speed scale 0..1')
    p.add_argument('--debug', '-d', action='store_true', help='show debug window')
    p.add_argument('--steps', action='store_true', help='show intermediate detection steps (edges/roi/hough)')
    p.add_argument('--save-steps', action='store_true', help='save intermediate detection images to debug_out/')
    p.add_argument('--sensitivity', type=float, default=0.8, help='how strongly forward speed reduces with steering error (0..1)')
    p.add_argument('--min-forward', type=float, default=0.3, help='minimum forward scale when turning (0..1)')
    p.add_argument('--view', choices=['left', 'center', 'right'], default='center', help='camera view preset to use (for tilted camera views)')
    p.add_argument('--calib', default=None, help='optional calibration .npz file with camera_matrix and dist_coefs')
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    try:
        video_source = int(args.video)
    except Exception:
        video_source = args.video
    # If the provided video source is a single image file, run one-shot
    # processing and save ROI/BEV/sliding-window/polynomial results to
    # lane_results/. Otherwise run the normal LaneFollower loop.
    if isinstance(video_source, str) and os.path.isfile(video_source):
        os.makedirs('lane_results', exist_ok=True)
        # load image and run detector debug
        frame = cv2.imread(video_source)
        if frame is None:
            logging.error('Failed to read image: %s', video_source)
            return
        # run a one-off detector pipeline
        detector = LaneDetection()
        try:
            detector.set_view(args.view)
        except Exception:
            pass
        # load calibration for one-shot if provided
        if args.calib:
            try:
                if os.path.isfile(args.calib):
                    npz = np.load(args.calib)
                    cam = npz.get('camera_matrix')
                    dist = npz.get('dist_coefs')
                    if cam is not None and dist is not None:
                        detector.set_camera_intrinsics(cam, dist)
            except Exception:
                logging.exception('Failed to load calibration file: %s', args.calib)
        # detect_with_debug returns (center, debug_dict)
        center, debug = detector.detect_with_debug(frame, is_rgb=False)
        # save debug intermediate images if present
        try:
            if 'proc' in debug and debug['proc'] is not None:
                cv2.imwrite('lane_results/proc.png', debug['proc'])
            if 'edges' in debug and debug['edges'] is not None:
                cv2.imwrite('lane_results/edges.png', debug['edges'])
            if 'roi' in debug and debug['roi'] is not None:
                cv2.imwrite('lane_results/roi.png', debug['roi'])
            if 'hough' in debug and debug['hough'] is not None:
                cv2.imwrite('lane_results/hough.png', debug['hough'])
        except Exception:
            logging.exception('Failed to save debug images')

        # compute BEV (bird's-eye view) from the ROI using the same
        # perspective matrices used by the detector
        try:
            if debug.get('roi') is not None:
                roi = debug['roi']
                h, w = roi.shape[:2]
                M, Minv = detector._get_perspective_matrices(w, h)
                proc = debug.get('proc')
                if proc is not None:
                    bev_proc = cv2.warpPerspective(proc, M, (w, h), flags=cv2.INTER_LINEAR)
                    cv2.imwrite('lane_results/bev_proc.png', bev_proc)
                bev_edges = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_NEAREST)
                cv2.imwrite('lane_results/bev_edges.png', bev_edges)

                # sliding-window visualization & fitted polynomial curves
                try:
                    warped = bev_edges
                    warped_proc = bev_proc if proc is not None else cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
                    h_w, w_w = warped.shape[:2]
                    hist = np.sum(warped[h_w // 2:, :], axis=0)
                    slide_vis = cv2.cvtColor((warped_proc).astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(warped_proc.shape) == 2 else warped_proc.copy()
                    lcoeff = None
                    rcoeff = None
                    if np.sum(hist) != 0:
                        midpoint = int(w_w // 2)
                        left_base = int(np.argmax(hist[:midpoint]))
                        right_base = int(np.argmax(hist[midpoint:]) + midpoint)
                        lx, ly = detector._sliding_window_search(warped, left_base)
                        rx, ry = detector._sliding_window_search(warped, right_base)
                        if lx.size > 0 and ly.size > 0:
                            lcoeff = detector._fit_polynomial(lx, ly, degree=2)
                            if lcoeff is not None:
                                ys = np.linspace(0, h_w - 1, h_w).astype(np.int32)
                                xs = np.polyval(lcoeff, ys).astype(np.int32)
                                for (xx, yy) in zip(xs, ys):
                                    if 0 <= xx < w_w and 0 <= yy < h_w:
                                        cv2.circle(slide_vis, (int(xx), int(yy)), 1, (0, 0, 255), -1)
                        if rx.size > 0 and ry.size > 0:
                            rcoeff = detector._fit_polynomial(rx, ry, degree=2)
                            if rcoeff is not None:
                                ys = np.linspace(0, h_w - 1, h_w).astype(np.int32)
                                xs = np.polyval(rcoeff, ys).astype(np.int32)
                                for (xx, yy) in zip(xs, ys):
                                    if 0 <= xx < w_w and 0 <= yy < h_w:
                                        cv2.circle(slide_vis, (int(xx), int(yy)), 1, (255, 0, 0), -1)
                    cv2.imwrite('lane_results/sliding_warped.png', slide_vis)

                    # save polynomial coefficients if available
                    with open('lane_results/polynomials.txt', 'w') as f:
                        f.write(f'left_coeff={None if lcoeff is None else list(map(float, lcoeff))}\n')
                        f.write(f'right_coeff={None if rcoeff is None else list(map(float, rcoeff))}\n')
                except Exception:
                    logging.exception('Failed to compute sliding-window visualization')
        except Exception:
            logging.exception('Failed to compute BEV/warped images')

        logging.info('Wrote lane results to lane_results/')
        return
    # otherwise run real-time lane follower
    lf = LaneFollower(video_source=video_source, base_speed=args.speed, steering_gain=args.gain, speed_scale=args.scale, speed_sensitivity=args.sensitivity, view=args.view, calib_path=args.calib)
    # pass step/debug flags into run so the run loop can display/save
    lf.run(show_debug=args.debug or args.steps)


if __name__ == '__main__':
    main()