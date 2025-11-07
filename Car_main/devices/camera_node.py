import cv2
import threading
import time
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Camera:
    """
    Simple USB camera handler using OpenCV VideoCapture.

    Features:
    - open/close camera
    - read single frames
    - optional background reader thread to always keep the latest frame
    - set/get resolution and fps
    - context-manager support
    """

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        backend: int = cv2.CAP_ANY,
        read_timeout: float = 1.0,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.read_timeout = read_timeout

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame = None  # type: Optional[Tuple[bool, any]]
        self._last_timestamp = 0.0

    def open(self) -> bool:
        """Open the camera device and apply settings (resolution, fps)."""
        if self._cap and self._cap.isOpened():
            return True
        self._cap = cv2.VideoCapture(self.index, self.backend)
        if not self._cap.isOpened():
            logger.error("Failed to open camera index %s", self.index)
            self._cap = None
            return False

        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        logger.info(
            "Opened camera %s (w=%s h=%s fps=%s)",
            self.index,
            self.get_width(),
            self.get_height(),
            self.get_fps(),
        )
        return True

    def close(self) -> None:
        """Stop background thread (if any) and release the capture."""
        self.stop()
        if self._cap:
            try:
                self._cap.release()
                logger.info("Camera %s released", self.index)
            except Exception:
                logger.exception("Error releasing camera")
            finally:
                self._cap = None

    def is_opened(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    def read(self, wait: bool = True) -> Tuple[bool, Optional[any]]:
        """
        Read a single frame. If a background reader is running, returns the latest frame.
        If not, reads directly from the VideoCapture. Returns (ret, frame).
        """
        if self._thread and self._thread.is_alive():
            # return latest frame from background thread
            timeout = time.time() + self.read_timeout if wait else time.time()
            while wait and time.time() < timeout:
                with self._lock:
                    if self._latest_frame is not None:
                        return self._latest_frame
                time.sleep(0.005)
            # timeout or not waiting
            with self._lock:
                return self._latest_frame if self._latest_frame is not None else (False, None)

        # direct read
        if not self.is_opened() and not self.open():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def read_rgb(self, wait: bool = True) -> Tuple[bool, Optional[any]]:
        """Read a frame and return it converted to RGB order.

        This wraps `read()` and performs a BGR->RGB conversion when a
        valid frame is available. It respects the same `wait` semantics
        as `read()` (i.e., will use the background reader if running).
        """
        ret, frame = self.read(wait=wait)
        if not ret or frame is None:
            return False, None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return True, rgb
        except Exception:
            # conversion failed; return original frame as a fallback
            return True, frame

    def start(self) -> bool:
        """Start a background reader thread that keeps the latest frame."""
        if self._running:
            return True
        if not self.is_opened() and not self.open():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info("Background camera thread started")
        return True

    def stop(self) -> None:
        """Stop background reader thread."""
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Background camera thread stopped")

    def _reader_loop(self) -> None:
        while self._running and self.is_opened():
            ret, frame = self._cap.read()
            ts = time.time()
            with self._lock:
                self._latest_frame = (ret, frame if ret else None)
                self._last_timestamp = ts
            # small sleep to avoid busy loop if camera can't supply fps
            time.sleep(0.001)
        # mark no frame available when loop ends
        with self._lock:
            self._latest_frame = None

    def get_frame(self) -> Tuple[bool, Optional[any]]:
        """Convenience to get the latest frame (non-blocking)."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return self._latest_frame

    def get_timestamp(self) -> float:
        with self._lock:
            return self._last_timestamp

    def set_resolution(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        if self.is_opened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    def set_fps(self, fps: int) -> None:
        self.fps = fps
        if self.is_opened():
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    def get_width(self) -> Optional[int]:
        if not self.is_opened():
            return self.width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> Optional[int]:
        if not self.is_opened():
            return self.height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self) -> Optional[float]:
        if not self.is_opened():
            return self.fps
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    # Context manager support
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":
    # small demo to test the camera class on a Linux machine
    cam = Camera(index=0, width=640, height=480, fps=30)
    if not cam.open():
        print("Failed to open camera 0")
    else:
        cam.start()
        start = time.time()
        try:
            while time.time() - start < 5:  # run for 5 seconds
                ret, frame = cam.read(wait=True)
                if not ret:
                    time.sleep(0.01)
                    continue
                # show a simple window for quick manual testing
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()