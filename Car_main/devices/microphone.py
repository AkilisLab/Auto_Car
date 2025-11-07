"""Microphone utilities for real-time recording.

Provides MicrophoneStream: a simple PyAudio-backed recorder that yields
numpy arrays of float32 audio frames suitable for passing to audio models.

Features:
- context manager (with MicrophoneStream(...) as mic:)
- iterator/generator via `generator()` that yields numpy arrays
- optional callback on each chunk
- start/stop, read() to get raw bytes or ndarray

This implementation prefers PyAudio (already available in project's venv).

Example:
    with MicrophoneStream(rate=16000, chunk=1024) as mic:
        for frames in mic.generator():
            # frames is a numpy float32 array with shape (chunk, channels)
            model.consume(frames)

"""
from typing import Callable, Generator, Optional
import threading
import numpy as np

try:
    import pyaudio
except Exception as e:
    raise ImportError("PyAudio is required for MicrophoneStream: install pyaudio") from e


class MicrophoneStream:
    """Real-time microphone stream using PyAudio.

    Yields numpy float32 arrays with values in range [-1.0, 1.0].

    Args:
        rate: sampling rate (Hz)
        channels: number of channels
        chunk: frames per buffer (number of samples per channel)
        format: PyAudio format (default: pyaudio.paInt16)
        device_index: optional device index for input device
        callback: optional callable called with (ndarray) on each chunk
    """

    def __init__(
        self,
        rate: int = 16000,
        channels: int = 1,
        chunk: int = 1024,
        format: int = pyaudio.paInt16,
        device_index: Optional[int] = None,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.format = format
        self.device_index = device_index
        self.callback = callback

        self._pa = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._lock = threading.Lock()
        self._running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def start(self) -> None:
        """Open and start the audio input stream."""
        with self._lock:
            if self._running:
                return
            self._stream = self._pa.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                input_device_index=self.device_index,
            )
            self._running = True

    def stop(self) -> None:
        """Stop and close the audio stream and terminate PyAudio if needed."""
        with self._lock:
            if not self._running:
                return
            try:
                if self._stream is not None:
                    self._stream.stop_stream()
                    self._stream.close()
            finally:
                self._stream = None
                self._running = False
                # Do NOT terminate the PyAudio instance here. Terminating
                # it makes restarting fail on some systems. Keep the
                # PyAudio instance alive so start() can reopen streams.
                return

    def close(self) -> None:
        """Terminate the underlying PyAudio instance. After this the
        MicrophoneStream instance can't be started again unless a new
        MicrophoneStream is created.
        """
        with self._lock:
            try:
                if self._stream is not None:
                    self._stream.stop_stream()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None
            self._running = False
            try:
                if self._pa is not None:
                    self._pa.terminate()
            except Exception:
                pass
            finally:
                # drop reference so a new PyAudio can be created if needed
                self._pa = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def read(self, as_float: bool = True) -> np.ndarray:
        """Read one chunk from the stream.

        Returns:
            ndarray: shape (chunk, channels) dtype float32 (if as_float) or int16
        """
        if not self._running or self._stream is None:
            raise RuntimeError("Stream is not running. Call start() or use context manager.")

        data = self._stream.read(self.chunk, exception_on_overflow=False)
        # convert bytes to numpy
        if self.format == pyaudio.paInt16:
            dtype = np.int16
            max_val = 32768.0
        elif self.format == pyaudio.paInt32:
            dtype = np.int32
            max_val = float(2 ** 31)
        elif self.format == pyaudio.paFloat32:
            dtype = np.float32
            max_val = 1.0
        else:
            # fallback
            dtype = np.int16
            max_val = 32768.0

        arr = np.frombuffer(data, dtype=dtype)
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        else:
            arr = arr.reshape(-1, 1)

        if as_float:
            if dtype == np.float32:
                out = arr.astype(np.float32)
            else:
                out = (arr.astype(np.float32) / max_val).astype(np.float32)
            return out
        return arr

    def generator(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields chunks as numpy float32 arrays.

        Yields:
            ndarray: shape (chunk, channels)
        """
        if not self._running:
            self.start()

        try:
            while self._running:
                chunk = self.read(as_float=True)
                if self.callback is not None:
                    try:
                        self.callback(chunk)
                    except Exception:
                        # don't let callback exceptions stop the stream
                        pass
                yield chunk
        finally:
            # do not auto-stop here — context manager or user should call stop()
            return

    def __iter__(self):
        return self.generator()

    def list_input_devices(self) -> list:
        """Return a list of (index, name) for available input devices."""
        devices = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append((i, info.get('name')))
        return devices


__all__ = ["MicrophoneStream"]
