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
        roi_width = int(width * 0.4)
        roi_height = int(height * 0.4)
        x0 = (width - roi_width) // 2
        y0 = height - roi_height
        return x0, y0, x0 + roi_width, y0 + roi_height

    @staticmethod
    def _point_in_roi(point: Tuple[int, int], roi: Tuple[int, int, int, int]) -> bool:
        x, y = point
        x0, y0, x1, y1 = roi
        return x0 <= x <= x1 and y0 <= y <= y1
