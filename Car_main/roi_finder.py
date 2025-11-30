#!/usr/bin/env python3
"""Simple ROI finder: open an image and print mouse-click coordinates.

Usage:
  python roi_finder.py [path/to/image.jpg]

Left-click: prints "x,y" to stdout and draws a small marker.
Press 's' to save collected points to `roi_points.txt` (one per line).
Press ESC or 'q' to quit.
"""

import argparse
import os
import sys
from typing import List, Tuple

import cv2


def main():
    p = argparse.ArgumentParser(description="Open image and print click coords")
    p.add_argument('image', nargs='?', default='snapshots/snap_20251109_121536.jpg',
                   help='Path to image (default: snapshots/snap_20251025_230641.jpg)')
    args = p.parse_args()

    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        sys.exit(2)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        sys.exit(2)

    window = 'ROI Finder'
    display = img.copy()
    points: List[Tuple[int, int]] = []

    def _draw_points():
        nonlocal display
        display = img.copy()
        for i, (x, y) in enumerate(points, start=1):
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def on_mouse(ev, x, y, flags, param):
        # Left button prints and marks the point
        if ev == cv2.EVENT_LBUTTONDOWN:
            print(f"{x},{y}")
            sys.stdout.flush()
            points.append((x, y))
            _draw_points()
            cv2.imshow(window, display)
        # Right button removes the last point (undo)
        elif ev == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                _draw_points()
                cv2.imshow(window, display)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    cv2.imshow(window, display)

    print("ROI finder: left-click to print coords, right-click to undo last, 's' to save, 'q' or ESC to quit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('s'):
            # save points to file
            try:
                out_path = 'roi_points.txt'
                with open(out_path, 'w') as f:
                    for x, y in points:
                        f.write(f"{x},{y}\n")
                print(f"Saved {len(points)} points -> {out_path}")
            except Exception as e:
                print("Failed to save points:", e)
        else:
            # ignore other keys; continue waiting
            continue

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
