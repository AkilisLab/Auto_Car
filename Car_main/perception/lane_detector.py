"""Lane detection utilities inspired by the AutomaticAddison OpenCV tutorial.

This implementation keeps the same external API expected by the rest of the
project (see `auto_mode.py`) while using a color+gradient thresholding
pipeline, perspective transform (bird's-eye view), sliding-window search and
polynomial fitting for lane extraction.

Public API (unchanged):
- detect_lane_center(frame, is_rgb=False) -> Optional[int]
- detect_with_debug(frame, is_rgb=False) -> Tuple[Optional[int], dict]

Private helpers preserved for use by `auto_mode.py` one-shot visualization:
- _sliding_window_search, _fit_polynomial, _get_perspective_matrices
"""
from typing import Optional, Tuple
import collections
import logging
import os
import time

import cv2
import numpy as np


class LaneDetection:
	"""Detects lane center from an RGB or BGR frame.

	The detection pipeline implemented here is:
	- color-space thresholds (HLS S-channel, R channel) + Sobel X on L
	- combine masks -> binary 'proc' image
	- apply trapezoidal ROI mask
	- compute perspective transform -> warp to bird's-eye view
	- histogram + sliding-window search to get lane pixels
	- fit 2nd-degree polynomials and compute lane center
	- temporal smoothing of center values
	"""

	def __init__(self, smoothing: int = 5):
		self.center_history = collections.deque(maxlen=smoothing)
		# view selection: 'left', 'center', 'right' (or custom)
		self.view = 'center'
		# small per-view rotation (degrees). Positive = CCW (OpenCV convention)
		# left should tilt CCW (positive), right should tilt CW (negative).
		self._view_rotations = {
			'center': 0.0,
			'left': 45.0,
			'right': -45.0,
		}

		# default view presets expressed as lists of four (x_rel, y_rel)
		# tuples for the perspective source trapezoid. Values are fractions
		# of width/height and are clipped to [0..1] when used.
		# make ROI flatter and bias toward the opposite side for tilted cameras
		# - center: slightly flatter
		# - left (camera turned left): focus more on right side (shift left bound rightwards,
		#   narrow top to the right)
		# - right (camera turned right): focus more on left side
		self._view_presets = {
			'center': [
				(0.0, 0.62),
				(1.0, 0.62),
				(0.85, 0.42),
				(0.15, 0.42),
			],
			# left camera (camera turned left): focus on right side -> taller & narrower
			'left': [
				(0.35, 0.62),  # bottom-left moved right to narrow base
				(1.0, 0.62),
				(0.95, 0.30),  # top-right high (taller)
				(0.70, 0.30),  # top-left moved toward center-right (narrow)
			],
			# right camera (camera turned right): focus on left side -> taller & narrower
			'right': [
				(0.0, 0.62),
				(0.65, 0.62),  # bottom-right moved left to narrow base
				(0.35, 0.30),  # top-right moved toward center-left (taller)
				(0.05, 0.30),  # top-left near far-left
			],
		}

		# optional camera intrinsics for undistortion: (camera_matrix, dist_coefs)
		self.cam_mtx = None
		self.dist_coefs = None
		# lazy undistort/remap caches
		self._undistort_maps = None  # tuple (map1, map2, map_size)

	def set_view_rotation(self, view_name: str, degrees: float) -> None:
		"""Set the rotation (degrees) applied for a named view preset.

		Positive angles rotate counter-clockwise (OpenCV convention).
		"""
		if view_name not in self._view_presets and view_name != 'center':
			raise ValueError('unknown view for rotation')
		self._view_rotations[view_name] = float(degrees)

	def set_view(self, view_name: str) -> None:
		"""Switch active view preset ('left'|'center'|'right')."""
		if view_name not in self._view_presets:
			raise ValueError(f"unknown view: {view_name}")
		self.view = view_name

	def set_view_points(self, points, name: str = 'custom') -> None:
		"""Register a custom view preset using 4 (x_rel,y_rel) points.

		`points` should be an iterable of four (x_rel, y_rel) pairs where each
		relative coordinate is in [0..1]. The preset will be stored under
		`name` and automatically selected.
		"""
		pts = list(points)
		if len(pts) != 4:
			raise ValueError('points must be length 4')
		# basic validation
		for p in pts:
			if not (isinstance(p, (list, tuple)) and len(p) == 2):
				raise ValueError('each point must be a tuple/list (x_rel,y_rel)')
		self._view_presets[name] = [(float(px), float(py)) for (px, py) in pts]
		self.view = name

	def set_camera_intrinsics(self, camera_matrix: np.ndarray, dist_coefs: np.ndarray) -> None:
		"""Provide camera matrix and distortion coefficients for undistortion."""
		self.cam_mtx = camera_matrix
		self.dist_coefs = dist_coefs
		# clear any previously computed maps so they'll be rebuilt with the
		# correct frame size when a frame is first provided to the detector.
		self._undistort_maps = None

	def _sliding_window_search(self, binary_img: np.ndarray, base_x: int,
						   nwindows: int = 9, margin: int = 100, minpix: int = 50):
		"""Sliding-window search to collect x,y pixels for one lane line.

		Returns (x, y) arrays (may be empty).
		"""
		if binary_img is None:
			return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
		nonzero = binary_img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		h, w = binary_img.shape
		x_current = base_x
		window_height = int(h // nwindows) if nwindows > 0 else h
		lane_inds = []
		for window in range(nwindows):
			win_y_low = h - (window + 1) * window_height
			win_y_high = h - window * window_height
			win_x_low = x_current - margin
			win_x_high = x_current + margin
			good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
					 (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
			if good_inds.size > 0:
				lane_inds.append(good_inds)
				# only recenter if enough pixels found
				if good_inds.size >= minpix:
					x_current = int(np.mean(nonzerox[good_inds]))

		if not lane_inds:
			return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
		lane_inds = np.concatenate(lane_inds)
		x = nonzerox[lane_inds]
		y = nonzeroy[lane_inds]
		return x, y

	def _fit_polynomial(self, x: np.ndarray, y: np.ndarray, degree: int = 2):
		"""Fit x = f(y) polynomial. Returns coeffs (highest-first) or None."""
		if x.size == 0 or y.size == 0:
			return None
		try:
			coeffs = np.polyfit(y, x, degree)
			return coeffs
		except Exception:
			logging.exception("polyfit failed")
			return None

	def _get_perspective_matrices(self, w: int, h: int):
		"""Return (M, Minv) for given ROI size and cache per-size.

		Source points chosen relatively to width/height to form a trapezoid.
		"""
		# include the active view in the cache key since source trapezoid may
		# differ between views and depend on rotation choices
		if hasattr(self, '_persp_cached') and self._persp_cached == (w, h, getattr(self, 'view', 'center')):
			return self._M, self._Minv

		# get relative trapezoid points for the active view preset
		rel = self._view_presets.get(getattr(self, 'view', 'center'))
		# clip and convert to absolute pixel coords
		rel_clipped = [(min(max(0.0, x), 1.0), min(max(0.0, y), 1.0)) for (x, y) in rel]
		src = np.float32([[w * x, h * y] for (x, y) in rel_clipped])

		dst = np.float32([
			[w * 0.20, h * 0.98],
			[w * 0.80, h * 0.98],
			[w * 0.80, 0.0],
			[w * 0.20, 0.0],
		])

		M = cv2.getPerspectiveTransform(src, dst)
		Minv = cv2.getPerspectiveTransform(dst, src)
		self._M = M
		self._Minv = Minv
		self._persp_cached = (w, h, getattr(self, 'view', 'center'))
		return M, Minv

	def _ensure_undistort_maps(self, frame_shape: Tuple[int, int]) -> None:
		"""Initialize undistort/remap maps for given frame shape (h, w).

		This is lazy: we only compute maps when we have the actual frame size.
		"""
		h, w = frame_shape
		if self.cam_mtx is None or self.dist_coefs is None:
			return
		if self._undistort_maps is None or self._undistort_maps[2] != (w, h):
			# compute maps
			new_mtx = self.cam_mtx
			map1, map2 = cv2.initUndistortRectifyMap(self.cam_mtx, self.dist_coefs, None, new_mtx, (w, h), cv2.CV_16SC2)
			self._undistort_maps = (map1, map2, (w, h))

	def _undistort_and_rotate(self, frame: np.ndarray) -> np.ndarray:
		"""Undistort (if intrinsics present) and apply per-view rotation.

		Returns a new frame (BGR ordering as input). Rotation is applied after
		undistortion. Rotation angle is taken from `_view_rotations` for the
		active view.
		"""
		if frame is None:
			return None
		img = frame
		# undistort if intrinsics provided
		if self.cam_mtx is not None and self.dist_coefs is not None:
			h, w = img.shape[:2]
			self._ensure_undistort_maps((h, w))
			if self._undistort_maps is not None:
				map1, map2, _ = self._undistort_maps
				try:
					img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
				except Exception:
					# fallback to cv2.undistort
					try:
						img = cv2.undistort(img, self.cam_mtx, self.dist_coefs, None, self.cam_mtx)
					except Exception:
						pass
		# apply per-view rotation if configured
		angle = float(self._view_rotations.get(getattr(self, 'view', 'center'), 0.0))
		if abs(angle) > 0.001:
			h, w = img.shape[:2]
			Mrot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
			# keep size same, use reflect border to avoid black corners
			img = cv2.warpAffine(img, Mrot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
		return img

	def _color_and_gradient_mask(self, frame: np.ndarray, is_rgb: bool = False) -> Tuple[np.ndarray, np.ndarray]:
		"""Compute combined binary mask from color thresholds and Sobel X.

		Returns (combined_mask_uint8, debug_proc) where combined_mask is 0/255
		and debug_proc is a grayscale image useful for saving/inspection.
		"""
		if frame is None:
			return None, None
		# convert to BGR ordering expected by OpenCV functions if needed
		img = frame.copy()
		if is_rgb:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		# Convert to HLS and extract channels
		hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
		h, l, s = cv2.split(hls)
		b, g, r = cv2.split(img)

		# Color thresholds (tuned defaults from the article)
		s_thresh = (80, 255)
		r_thresh = (120, 255)
		s_binary = np.zeros_like(s, dtype=np.uint8)
		s_binary[(s >= s_thresh[0]) & (s <= s_thresh[1])] = 255
		r_binary = np.zeros_like(r, dtype=np.uint8)
		r_binary[(r >= r_thresh[0]) & (r <= r_thresh[1])] = 255

		# Sobel X on L channel (detect vertical-ish edges)
		# convert to uint8 if needed
		l_u = l if l.dtype == np.uint8 else (l * 255).astype(np.uint8)
		sobelx = cv2.Sobel(l_u, cv2.CV_64F, 1, 0, ksize=3)
		abs_sobelx = np.absolute(sobelx)
		scaled = np.uint8(255 * abs_sobelx / (np.max(abs_sobelx) + 1e-6))
		# Sobel thresholds (tunable)
		sobel_thresh = (20, 100)
		sobel_binary = np.zeros_like(scaled, dtype=np.uint8)
		sobel_binary[(scaled >= sobel_thresh[0]) & (scaled <= sobel_thresh[1])] = 255

		# Combine: color OR sobel (article uses combinations; this is a robust start)
		combined = np.zeros_like(s_binary, dtype=np.uint8)
		combined[((s_binary == 255) & (r_binary == 255)) | (sobel_binary == 255)] = 255

		# debug_proc: for visualization keep a stacked view (grayscale)
		# we return combined (binary) as proc for compatibility
		return combined, combined

	def region_of_interest(self, binary: np.ndarray) -> np.ndarray:
		"""Mask binary image with trapezoidal ROI aligned with perspective src.

		Return masked binary image (same shape).
		"""
		if binary is None:
			return None
		h, w = binary.shape
		# build polygon from active view preset for consistency with perspective src
		rel = self._view_presets.get(getattr(self, 'view', 'center'))
		rel_clipped = [(min(max(0.0, x), 1.0), min(max(0.0, y), 1.0)) for (x, y) in rel]
		src_poly = np.array([[ (int(w * x), int(h * y)) for (x, y) in rel_clipped ]], dtype=np.int32)
		mask = np.zeros_like(binary)
		cv2.fillPoly(mask, src_poly, 255)
		return cv2.bitwise_and(binary, mask)

	def detect_lane_center(self, frame: np.ndarray, is_rgb: bool = False) -> Optional[int]:
		center, _ = self.detect_with_debug(frame, is_rgb=is_rgb)
		return center

	def detect_with_debug(self, frame: np.ndarray, is_rgb: bool = False) -> Tuple[Optional[int], dict]:
		"""Detect lane center and return debug images.

		debug dict keys: 'proc' (binary mask), 'edges' (sobel/grad), 'roi' (masked binary), 'hough' (BGR overlay).
		"""
		debug = {}
		# Step 0: undistort (if intrinsics present) and always apply per-view rotation
		undistorted = self._undistort_and_rotate(frame)
		debug['undistorted'] = undistorted

		proc, proc_dbg = self._color_and_gradient_mask(undistorted, is_rgb=is_rgb)
		debug['proc'] = proc_dbg
		if proc is None:
			return None, debug

		roi = self.region_of_interest(proc)
		debug['roi'] = roi
		if roi is None:
			return None, debug

		h, w = roi.shape
		# perspective warp to bird's-eye
		try:
			M, Minv = self._get_perspective_matrices(w, h)
			warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_NEAREST)
		except Exception:
			warped = roi.copy()

		# histogram over lower half
		hist = np.sum(warped[h // 2:, :], axis=0)

		# prepare visualization in warped coords (BGR)
		hough_vis_warped = cv2.cvtColor((warped).astype(np.uint8), cv2.COLOR_GRAY2BGR)
		left_x = None
		right_x = None
		lcoeff = None
		rcoeff = None

		if np.sum(hist) == 0:
			# fallback to Hough on roi (original view) for visualization
			lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=40,
						minLineLength=40, maxLineGap=30)
			hough_vis = cv2.cvtColor(proc_dbg, cv2.COLOR_GRAY2BGR) if proc_dbg is not None else None
			if lines is not None and hough_vis is not None:
				for l in lines.reshape(-1, 4):
					x1, y1, x2, y2 = l
					cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
					if x2 == x1:
						continue
					slope = (y2 - y1) / (x2 - x1 + 1e-6)
					if abs(slope) < 0.3:
						continue
					intercept = y1 - slope * x1
					if slope < 0:
						left_x = int((h - intercept) / (slope + 1e-6))
					else:
						right_x = int((h - intercept) / (slope + 1e-6))
			debug['hough'] = hough_vis

		# persist intermediate images for offline inspection in lane_results/
		try:
			os.makedirs('lane_results', exist_ok=True)
			ts = int(time.time() * 1000)
			if debug.get('undistorted') is not None:
				try:
					cv2.imwrite(f'lane_results/undistorted_{ts}.png', debug.get('undistorted'))
				except Exception:
					pass
			if debug.get('proc') is not None:
				try:
					cv2.imwrite(f'lane_results/proc_{ts}.png', debug.get('proc'))
				except Exception:
					pass
			if debug.get('roi') is not None:
				try:
					cv2.imwrite(f'lane_results/roi_{ts}.png', debug.get('roi'))
				except Exception:
					pass
			if debug.get('hough') is not None:
				try:
					cv2.imwrite(f'lane_results/hough_{ts}.png', debug.get('hough'))
				except Exception:
					pass
		except Exception:
			logging.exception('Failed to write lane_results images')
		else:
			midpoint = int(w // 2)
			left_base = int(np.argmax(hist[:midpoint])) if midpoint > 0 else 0
			right_base = int(np.argmax(hist[midpoint:]) + midpoint) if midpoint > 0 else 0
			# perform sliding-window on warped image
			lx, ly = self._sliding_window_search(warped, left_base)
			rx, ry = self._sliding_window_search(warped, right_base)
			if lx.size > 0 and ly.size > 0:
				lcoeff = self._fit_polynomial(lx, ly, degree=2)
				if lcoeff is not None:
					left_x = int(np.polyval(lcoeff, h - 1))
					ys = np.linspace(0, h - 1, h).astype(np.int32)
					xs = np.polyval(lcoeff, ys).astype(np.int32)
					for (xx, yy) in zip(xs, ys):
						if 0 <= xx < w and 0 <= yy < h:
							hough_vis_warped = cv2.circle(hough_vis_warped, (int(xx), int(yy)), 1, (0, 0, 255), -1)
			if rx.size > 0 and ry.size > 0:
				rcoeff = self._fit_polynomial(rx, ry, degree=2)
				if rcoeff is not None:
					right_x = int(np.polyval(rcoeff, h - 1))
					ys = np.linspace(0, h - 1, h).astype(np.int32)
					xs = np.polyval(rcoeff, ys).astype(np.int32)
					for (xx, yy) in zip(xs, ys):
						if 0 <= xx < w and 0 <= yy < h:
							hough_vis_warped = cv2.circle(hough_vis_warped, (int(xx), int(yy)), 1, (255, 0, 0), -1)
			# warp visualization back to original view
			try:
				hough_vis = cv2.warpPerspective(hough_vis_warped, Minv, (w, h), flags=cv2.INTER_LINEAR)
			except Exception:
				hough_vis = cv2.cvtColor(proc_dbg, cv2.COLOR_GRAY2BGR)
			debug['hough'] = hough_vis

		# compute center in warped coords
		center = None
		if left_x is not None and right_x is not None:
			center_warp = (left_x + right_x) // 2
		elif left_x is not None:
			estimated_right = left_x + int(w * 0.4)
			center_warp = (left_x + estimated_right) // 2
		elif right_x is not None:
			estimated_left = right_x - int(w * 0.4)
			center_warp = (estimated_left + right_x) // 2
		else:
			center_warp = None

		if center_warp is None:
			return None, debug

		# map center back to original image coordinates
		try:
			pt = np.array([[[center_warp, h - 1]]], dtype=np.float32)
			orig_pt = cv2.perspectiveTransform(pt, Minv)[0][0]
			center_orig = int(orig_pt[0])
		except Exception:
			center_orig = int(center_warp)

		# append and return smoothed center in original image space
		try:
			self.center_history.append(center_orig)
			smooth = int(np.mean(self.center_history))
			return smooth, debug
		except Exception:
			return center_orig, debug
		

# """Lane detection utilities extracted from `auto_mode.py`.

# This module provides the `LaneDetection` class which exposes two
# public methods used by the rest of the project:
# - `detect_lane_center(frame, is_rgb=False) -> Optional[int]`
# - `detect_with_debug(frame, is_rgb=False) -> Tuple[Optional[int], dict]`

# The implementation is intentionally lightweight: grayscale -> blur ->
# Canny -> either sliding-window (BEV) or Hough fallback.
# """
# from typing import Optional, Tuple
# import collections
# import logging

# import cv2
# import numpy as np


# class LaneDetection:
# 	"""Detects lane center from an RGB or BGR frame.

# 	The detection API accepts frames in either color order; the caller
# 	should indicate which ordering is active via `is_rgb`.
# 	"""

# 	def __init__(self, smoothing: int = 5):
# 		self.center_history = collections.deque(maxlen=smoothing)

# 	def _sliding_window_search(self, binary_img: np.ndarray, base_x: int,
# 							   nwindows: int = 9, margin: int = 100, minpix: int = 50):
# 		"""Perform sliding window search on a binary (edges) image for one lane line.

# 		Returns arrays of x and y pixel coordinates for the detected line (may be empty).
# 		"""
# 		if binary_img is None:
# 			return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
# 		# nonzero pixel indices
# 		nonzero = binary_img.nonzero()
# 		nonzeroy = np.array(nonzero[0])
# 		nonzerox = np.array(nonzero[1])
# 		h, w = binary_img.shape
# 		# current x center
# 		x_current = base_x
# 		window_height = int(h // nwindows)
# 		lane_inds = []
# 		for window in range(nwindows):
# 			win_y_low = h - (window + 1) * window_height
# 			win_y_high = h - window * window_height
# 			win_x_low = x_current - margin
# 			win_x_high = x_current + margin
# 			good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
# 						 (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
# 			if good_inds.size > 0:
# 				lane_inds.append(good_inds)
# 				# recenter to mean of detected pixels
# 				x_current = int(np.mean(nonzerox[good_inds]))

# 		if not lane_inds:
# 			return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
# 		lane_inds = np.concatenate(lane_inds)
# 		x = nonzerox[lane_inds]
# 		y = nonzeroy[lane_inds]
# 		return x, y

# 	def _fit_polynomial(self, x: np.ndarray, y: np.ndarray, degree: int = 2):
# 		"""Fit a polynomial x = f(y) of given degree. Returns coefficients or None."""
# 		if x.size == 0 or y.size == 0:
# 			return None
# 		try:
# 			coeffs = np.polyfit(y, x, degree)
# 			return coeffs
# 		except Exception:
# 			logging.exception("polyfit failed")
# 			return None

# 	def _get_perspective_matrices(self, w: int, h: int):
# 		"""Return (M, Minv) perspective transform matrices for given frame size.

# 		The source trapezoid is chosen relative to the image size and maps to
# 		a rectangular bird's-eye view. Matrices are cached per-instance for
# 		the most recent size.
# 		"""
# 		# cache last
# 		if hasattr(self, '_persp_cached') and self._persp_cached == (w, h):
# 			return self._M, self._Minv

# 		src = np.float32([
# 			[w * 0, h * 0.55],
# 			[w * 1, h * 0.55],
# 			[w * 0.8, h * 0.25],
# 			[w * 0.2, h * 0.25],
# 		])

# 		dst = np.float32([
# 			[w * 0.20, h * 0.98],
# 			[w * 0.80, h * 0.98],
# 			[w * 0.80, 0.0],
# 			[w * 0.20, 0.0],
# 		])

# 		M = cv2.getPerspectiveTransform(src, dst)
# 		Minv = cv2.getPerspectiveTransform(dst, src)
# 		self._M = M
# 		self._Minv = Minv
# 		self._persp_cached = (w, h)
# 		return M, Minv

# 	def preprocess(self, frame: np.ndarray, is_rgb: bool = False) -> np.ndarray:
# 		if frame is None:
# 			return None
# 		if is_rgb:
# 			gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# 		else:
# 			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 		blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 		return blur

# 	def edges(self, img: np.ndarray) -> np.ndarray:
# 		v = np.median(img)
# 		lower = int(max(0, 0.66 * v))
# 		upper = int(min(255, 1.33 * v))
# 		return cv2.Canny(img, lower, upper)

# 	def region_of_interest(self, edges: np.ndarray) -> np.ndarray:
# 		h, w = edges.shape
# 		mask = np.zeros_like(edges)
# 		polygon = np.array([[
# 			(0, h),
# 			(0, int(h * 0.6)),
# 			(w, int(h * 0.6)),
# 			(w, h),
# 		]], dtype=np.int32)
# 		cv2.fillPoly(mask, polygon, 255)
# 		return cv2.bitwise_and(edges, mask)

# 	def detect_lane_center(self, frame: np.ndarray, is_rgb: bool = False) -> Optional[int]:
# 		proc = self.preprocess(frame, is_rgb=is_rgb)
# 		if proc is None:
# 			return None
# 		ed = self.edges(proc)
# 		roi = self.region_of_interest(ed)
# 		if roi is None:
# 			return None
# 		h, w = roi.shape

# 		# compute perspective transform and warp ROI to bird's-eye view
# 		try:
# 			M, Minv = self._get_perspective_matrices(w, h)
# 			warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_LINEAR)
# 		except Exception:
# 			warped = roi.copy()

# 		# Sliding-window approach based on histogram of lower half (on warped image)
# 		hist = np.sum(warped[h // 2:, :], axis=0)
# 		if np.sum(hist) == 0:
# 			# no strong signal, fallback to old Hough-based approach
# 			lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=40,
# 									minLineLength=40, maxLineGap=30)
# 			if lines is None:
# 				return None
# 			left_lines = []
# 			right_lines = []
# 			for l in lines.reshape(-1, 4):
# 				x1, y1, x2, y2 = l
# 				if x2 == x1:
# 					continue
# 				slope = (y2 - y1) / (x2 - x1 + 1e-6)
# 				if abs(slope) < 0.3:
# 					continue
# 				intercept = y1 - slope * x1
# 				if slope < 0:
# 					left_lines.append((slope, intercept))
# 				else:
# 					right_lines.append((slope, intercept))

# 			def extrapolate(params):
# 				if not params:
# 					return None
# 				slope_avg = np.mean([p[0] for p in params])
# 				intercept_avg = np.mean([p[1] for p in params])
# 				return int((h - intercept_avg) / (slope_avg + 1e-6))

# 			left_x = extrapolate(left_lines)
# 			right_x = extrapolate(right_lines)
# 		else:
# 			midpoint = int(w // 2)
# 			left_base = int(np.argmax(hist[:midpoint]))
# 			right_base = int(np.argmax(hist[midpoint:]) + midpoint)
# 			# perform sliding-window on warped image
# 			lx, ly = self._sliding_window_search(warped, left_base)
# 			rx, ry = self._sliding_window_search(warped, right_base)

# 			left_x = None
# 			right_x = None
# 			if lx.size > 0 and ly.size > 0:
# 				lcoeff = self._fit_polynomial(lx, ly, degree=2)
# 				if lcoeff is not None:
# 					# evaluate at bottom (y = h) in warped coords
# 					left_x = int(np.polyval(lcoeff, h - 1))
# 			if rx.size > 0 and ry.size > 0:
# 				rcoeff = self._fit_polynomial(rx, ry, degree=2)
# 				if rcoeff is not None:
# 					right_x = int(np.polyval(rcoeff, h - 1))

# 		# compute center and apply temporal smoothing
# 		center = None
# 		if left_x is not None and right_x is not None:
# 			center = (left_x + right_x) // 2
# 		elif left_x is not None:
# 			estimated_right = left_x + int(w * 0.4)
# 			center = (left_x + estimated_right) // 2
# 		elif right_x is not None:
# 			estimated_left = right_x - int(w * 0.4)
# 			center = (estimated_left + right_x) // 2

# 		if center is None:
# 			return None

# 		# map center (in warped coords) back to original image coordinates
# 		try:
# 			pt = np.array([[[center, h - 1]]], dtype=np.float32)
# 			orig_pt = cv2.perspectiveTransform(pt, Minv)[0][0]
# 			center_orig = int(orig_pt[0])
# 		except Exception:
# 			center_orig = int(center)

# 		# append and return smoothed center in original image space
# 		try:
# 			self.center_history.append(center_orig)
# 			smooth = int(np.mean(self.center_history))
# 			return smooth
# 		except Exception:
# 			return center_orig

# 	def detect_with_debug(self, frame: np.ndarray, is_rgb: bool = False) -> Tuple[Optional[int], dict]:
# 		"""Detect lane center and also return intermediate debug images.

# 		Returns (center, debug_dict) where debug_dict may contain keys:
# 		'proc' (blurred gray), 'edges', 'roi', 'hough' (BGR image with lines drawn).
# 		"""
# 		debug = {}
# 		proc = self.preprocess(frame, is_rgb=is_rgb)
# 		debug['proc'] = proc
# 		if proc is None:
# 			return None, debug
# 		ed = self.edges(proc)
# 		debug['edges'] = ed
# 		roi = self.region_of_interest(ed)
# 		debug['roi'] = roi
# 		if roi is None:
# 			return None, debug
# 		h, w = roi.shape

# 		# Try perspective warp -> sliding window approach first (preferred)
# 		try:
# 			M, Minv = self._get_perspective_matrices(w, h)
# 			warped_proc = cv2.warpPerspective(proc, M, (w, h), flags=cv2.INTER_LINEAR)
# 			warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_NEAREST)
# 		except Exception:
# 			warped_proc = proc.copy()
# 			warped = roi.copy()

# 		hist = np.sum(warped[h // 2:, :], axis=0)
# 		# build visualization in warped space, then warp back for display
# 		hough_vis_warped = cv2.cvtColor((warped_proc).astype(np.uint8), cv2.COLOR_GRAY2BGR)
# 		left_x = None
# 		right_x = None
# 		if np.sum(hist) == 0:
# 			# fallback to Hough for visualization in original view
# 			lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=40,
# 									minLineLength=40, maxLineGap=30)
# 			hough_vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
# 			if lines is not None:
# 				for l in lines.reshape(-1, 4):
# 					x1, y1, x2, y2 = l
# 					cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 					if x2 == x1:
# 						continue
# 					slope = (y2 - y1) / (x2 - x1 + 1e-6)
# 					if abs(slope) < 0.3:
# 						continue
# 					intercept = y1 - slope * x1
# 					if slope < 0:
# 						left_x = int((h - intercept) / (slope + 1e-6))
# 					else:
# 						right_x = int((h - intercept) / (slope + 1e-6))
# 			debug['hough'] = hough_vis
# 		else:
# 			midpoint = int(w // 2)
# 			left_base = int(np.argmax(hist[:midpoint]))
# 			right_base = int(np.argmax(hist[midpoint:]) + midpoint)
# 			lx, ly = self._sliding_window_search(warped, left_base)
# 			rx, ry = self._sliding_window_search(warped, right_base)
# 			if lx.size > 0 and ly.size > 0:
# 				lcoeff = self._fit_polynomial(lx, ly, degree=2)
# 				if lcoeff is not None:
# 					left_x = int(np.polyval(lcoeff, h - 1))
# 					ys = np.linspace(0, h - 1, h).astype(np.int32)
# 					xs = np.polyval(lcoeff, ys).astype(np.int32)
# 					for (xx, yy) in zip(xs, ys):
# 						if 0 <= xx < w and 0 <= yy < h:
# 							hough_vis_warped = cv2.circle(hough_vis_warped, (int(xx), int(yy)), 1, (0, 0, 255), -1)
# 			if rx.size > 0 and ry.size > 0:
# 				rcoeff = self._fit_polynomial(rx, ry, degree=2)
# 				if rcoeff is not None:
# 					right_x = int(np.polyval(rcoeff, h - 1))
# 					ys = np.linspace(0, h - 1, h).astype(np.int32)
# 					xs = np.polyval(rcoeff, ys).astype(np.int32)
# 					for (xx, yy) in zip(xs, ys):
# 						if 0 <= xx < w and 0 <= yy < h:
# 							hough_vis_warped = cv2.circle(hough_vis_warped, (int(xx), int(yy)), 1, (255, 0, 0), -1)
# 			# warp visualization back to original view
# 			try:
# 				hough_vis = cv2.warpPerspective(hough_vis_warped, Minv, (w, h), flags=cv2.INTER_LINEAR)
# 			except Exception:
# 				hough_vis = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
# 			debug['hough'] = hough_vis

# 		center = None
# 		if left_x is not None and right_x is not None:
# 			center_warp = (left_x + right_x) // 2
# 		elif left_x is not None:
# 			estimated_right = left_x + int(w * 0.4)
# 			center_warp = (left_x + estimated_right) // 2
# 		elif right_x is not None:
# 			estimated_left = right_x - int(w * 0.4)
# 			center_warp = (estimated_left + right_x) // 2
# 		else:
# 			center_warp = None

# 		if center_warp is None:
# 			return None, debug

# 		# map center back to original image coordinates
# 		try:
# 			pt = np.array([[[center_warp, h - 1]]], dtype=np.float32)
# 			orig_pt = cv2.perspectiveTransform(pt, Minv)[0][0]
# 			center_orig = int(orig_pt[0])
# 		except Exception:
# 			center_orig = int(center_warp)

# 		try:
# 			self.center_history.append(center_orig)
# 			smooth = int(np.mean(self.center_history))
# 			return smooth, debug
# 		except Exception:
# 			return center_orig, debug

