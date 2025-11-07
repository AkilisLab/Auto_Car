import threading
import logging
import time
import io
from typing import Optional, List, Dict, Any

try:
    from pydub import AudioSegment, playback
except Exception:
    AudioSegment = None
    playback = None

# optional simpleaudio to allow stopping playback mid-stream
try:
    import simpleaudio as sa
except Exception:
    sa = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Speaker:
    """
    Audio output handler using pydub only.

    Notes:
    - Decodes WAV/MP3/OGG/etc via pydub (ffmpeg required for many formats).
    - Playback uses simpleaudio (if available) for interruptible playback, otherwise pydub.playback.
    - Non-blocking playback is implemented by a thread.
    - device_index is accepted but ignored: pydub does not provide device selection.
    """

    def __init__(self, device_index: Optional[int] = None):
        if AudioSegment is None or playback is None:
            raise RuntimeError("pydub is required (pip install pydub) and ffmpeg must be available")
        self.device_index = device_index
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._playing = False
        # Whether simpleaudio is available for stoppable playback
        self._use_simpleaudio = sa is not None

    @staticmethod
    def list_output_devices() -> List[Dict]:
        """
        pydub does not enumerate output devices. Return a minimal hint list.
        For device selection use a different backend (not available when using only pydub).
        """
        return [{"index": None, "name": "system-default", "note": "pydub does not support device selection"}]

    def is_playing(self) -> bool:
        return self._playing

    def _play_with_simpleaudio(self, seg: Any, loop: bool):
        """
        Play using simpleaudio.play_buffer which returns a PlayObject that can be stopped.
        """
        try:
            raw = seg.raw_data
            channels = seg.channels
            sample_width = seg.sample_width  # bytes per sample
            frame_rate = seg.frame_rate

            self._playing = True
            while not self._stop_event.is_set():
                play_obj = sa.play_buffer(raw, channels, sample_width, frame_rate)
                # wait while playing, but allow stop_event to interrupt
                while play_obj.is_playing():
                    if self._stop_event.is_set():
                        try:
                            play_obj.stop()
                        except Exception:
                            pass
                        break
                    time.sleep(0.05)
                if not loop:
                    break
        except Exception:
            logger.exception("Playback (simpleaudio) error")
        finally:
            self._playing = False

    def _play_segment(self, seg: Any, loop: bool):
        try:
            # prefer simpleaudio if available (interruptible)
            if self._use_simpleaudio:
                self._play_with_simpleaudio(seg, loop)
                return

            # fallback: use pydub.playback (may not be interruptible)
            self._playing = True
            while not self._stop_event.is_set():
                playback.play(seg)
                if not loop:
                    break
        except Exception:
            logger.exception("Playback error")
        finally:
            self._playing = False

    def play_wav(self, filepath: str, loop: bool = False, blocking: bool = True) -> bool:
        """
        Play audio file (wav, mp3, ogg, ...). Uses pydub for decoding.
        """
        if AudioSegment is None:
            logger.error("pydub not available")
            return False

        try:
            seg = AudioSegment.from_file(filepath)
        except Exception:
            logger.exception("Failed to load audio file: %s", filepath)
            return False

        # start playback
        self._stop_event.clear()
        if blocking:
            self._play_segment(seg, loop=loop)
            return True

        # non-blocking
        with self._lock:
            self._play_thread = threading.Thread(target=self._play_segment, args=(seg, loop), daemon=True)
            self._play_thread.start()
        return True

    def play_frames(self, data: bytes, sample_width: int, frame_rate: int, channels: int, loop: bool = False, blocking: bool = True) -> bool:
        """
        Play raw PCM bytes. Provide sample_width (bytes per sample), frame_rate and channels.
        """
        if AudioSegment is None:
            logger.error("pydub not available")
            return False

        try:
            bio = io.BytesIO(data)
            seg = AudioSegment.from_raw(bio, sample_width=sample_width, frame_rate=frame_rate, channels=channels)
        except Exception:
            logger.exception("Failed to create AudioSegment from raw bytes")
            return False

        self._stop_event.clear()
        if blocking:
            self._play_segment(seg, loop=loop)
            return True

        with self._lock:
            self._play_thread = threading.Thread(target=self._play_segment, args=(seg, loop), daemon=True)
            self._play_thread.start()
        return True

    def stop(self) -> None:
        """
        Stop non-blocking playback. If simpleaudio is available, this will interrupt playback quickly.
        Otherwise, it will prevent subsequent loops but may not stop an in-flight playback immediately.
        """
        self._stop_event.set()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None
        self._playing = False

    def close(self) -> None:
        """Alias to stop for API parity with other backends."""
        self.stop()

    # context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


if __name__ == "__main__":
    # quick demo: lists devices and plays a file (pydub + ffmpeg required for formats like mp3)
    if AudioSegment is None:
        print("pydub not installed. Install with: pip install pydub and ensure ffmpeg is available")
    else:
        print("Output devices (hint):", Speaker.list_output_devices())
        sp = Speaker()
        sp.play_wav("./jazz.mp3", blocking=True)
        sp.close()