"""ArUco marker detection helpers for pre-acquired frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import cv2.aruco as aruco

LOG = logging.getLogger(__name__)


@dataclass
class DetectedMarker:
    marker_id: int
    corners: List[Tuple[int, int]]
    center: Tuple[int, int]
    confidence: float


class ArUcoDetector:
    """Detect ArUco markers on demand from provided frames."""

    def __init__(
        self,
        dictionary: int = aruco.DICT_4X4_50,
        show_debug: bool = False,
    ) -> None:
        self.show_debug = show_debug
        self._aruco_dict = aruco.getPredefinedDictionary(dictionary)
        self._parameters = aruco.DetectorParameters()
        self._detector = aruco.ArucoDetector(self._aruco_dict, self._parameters)

    def detect(self, frame) -> List[DetectedMarker]:
        if frame is None:
            return []
        height, width = frame.shape[:2]
        roi = self._bottom_center_roi(width, height)
        corners, ids, rejected = self._detector.detectMarkers(frame)
        markers: List[DetectedMarker] = []
        if ids is None:
            if self.show_debug:
                self._render_debug(frame, corners, ids, rejected)
            return markers
        for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
            flat = marker_corners.reshape(-1, 2)
            points = [(int(x), int(y)) for x, y in flat]
            center = self._compute_center(points)
            if not self._point_in_roi(center, roi):
                LOG.debug("Marker %s outside ROI, skipping", marker_id)
                continue
            markers.append(
                DetectedMarker(
                    marker_id=marker_id,
                    corners=points,
                    center=center,
                    confidence=self._estimate_confidence(flat),
                )
            )
        if self.show_debug:
            self._render_debug(frame, corners, ids, rejected)
        return markers

    def _render_debug(self, frame, corners, ids, rejected) -> None:
        debug_img = frame.copy()
        aruco.drawDetectedMarkers(debug_img, corners, ids)
        if rejected:
            aruco.drawDetectedMarkers(debug_img, rejected, borderColor=(0, 0, 255))
        cv2.imshow("ArUco Detector", debug_img)

    @staticmethod
    def _compute_center(points: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not points:
            return 0, 0
        xs = sum(pt[0] for pt in points)
        ys = sum(pt[1] for pt in points)
        count = len(points)
        return xs // count, ys // count

    @staticmethod
    def _estimate_confidence(points) -> float:
        if points is None:
            return 0.0
        side_lengths = []
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            side_lengths.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
        if not side_lengths:
            return 0.0
        min_side = min(side_lengths)
        max_side = max(side_lengths)
        if max_side == 0:
            return 0.0
        return min_side / max_side

    @staticmethod
    def _bottom_center_roi(width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Return an ROI rectangle expected to contain the ArUco marker.

        The user provided a tight bounding polygon in counter-clockwise order
        (top-left first) for the usual camera resolution of 640x480:
            (111,89), (144,386), (448,375), (461,75)

        To support different camera resolutions, scale those base coordinates
        to the requested `width`/`height`. If the image matches 640x480 the
        returned rectangle will be exactly the tight bounds for the marker.
        """
        # Base coordinates (reference resolution 640x480)
        BASE_W, BASE_H = 640, 480
        base_coords = [(111, 89), (144, 386), (448, 375), (461, 75)]

        if width == BASE_W and height == BASE_H:
            xs = [p[0] for p in base_coords]
            ys = [p[1] for p in base_coords]
        else:
            sx = float(width) / float(BASE_W)
            sy = float(height) / float(BASE_H)
            xs = [int(p[0] * sx) for p in base_coords]
            ys = [int(p[1] * sy) for p in base_coords]

        x0 = min(xs)
        y0 = min(ys)
        x1 = max(xs)
        y1 = max(ys)
        # Clamp to image bounds
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))
        return x0, y0, x1, y1

    @staticmethod
    def _point_in_roi(point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        x, y = point
        x0, y0, x1, y1 = roi
        return x0 <= x <= x1 and y0 <= y <= y1
