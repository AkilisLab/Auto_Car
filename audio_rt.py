"""Simple recorder CLI that toggles microphone recording with the spacebar.

Press SPACE to start recording, SPACE again to stop and write a WAV file.
Press Ctrl-C to exit.

This script uses `devices.microphone.MicrophoneStream`.
"""

import threading
import time
import wave
import os
from datetime import datetime
from typing import List

import numpy as np

from devices.microphone import MicrophoneStream


def _write_wav(path: str, frames: List[np.ndarray], rate: int, channels: int) -> None:
	"""Write list of float32 numpy arrays to a WAV file (int16)."""
	if not frames:
		return
	# concatenate
	data = np.vstack(frames)
	# clamp
	data = np.clip(data, -1.0, 1.0)
	# convert to int16
	ints = (data * 32767.0).astype(np.int16)
	# if multi-channel, ensure shape matches
	if channels == 1:
		raw = ints.flatten().tobytes()
	else:
		raw = ints.tobytes()

	os.makedirs(os.path.dirname(path), exist_ok=True)
	wf = wave.open(path, 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(2)  # 2 bytes for int16
	wf.setframerate(rate)
	wf.writeframes(raw)
	wf.close()


def _key_listener(toggle_fn):
	"""Listen for single-space keypresses on Linux terminal to toggle recording.

	This uses termios to read single characters without Enter.
	"""
	import sys
	import termios
	import tty

	fd = sys.stdin.fileno()
	old = termios.tcgetattr(fd)
	try:
		tty.setcbreak(fd)
		while True:
			ch = sys.stdin.read(1)
			# If stdin closed, sys.stdin.read will return empty string; exit loop
			if ch == '':
				break
			if ch == ' ':
				toggle_fn()
	except KeyboardInterrupt:
		return
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
	rate = 16000
	channels = 1
	chunk = 1024

	mic = MicrophoneStream(rate=rate, channels=channels, chunk=chunk)

	recording = {'on': False}
	frames: List[np.ndarray] = []
	stream_thread = None
	stop_event = threading.Event()


	def start_recording():
		nonlocal stream_thread, frames
		frames = []
		# Try to start; if opening the default device fails, try
		# selecting the first available input device.
		try:
			mic.start()
		except Exception:
			# try to pick a device
			try:
				devices = mic.list_input_devices()
				if devices:
					idx, name = devices[0]
					print(f"Falling back to input device {idx}: {name}")
					mic.device_index = idx
					mic.start()
				else:
					raise
			except Exception as e:
				print("Failed to open audio input device:", e)
				return
		stop_event.clear()

		def run():
			# read until stop_event is set
			while not stop_event.is_set():
				try:
					chunk_arr = mic.read(as_float=True)
				except RuntimeError:
					break
				frames.append(chunk_arr)

		stream_thread = threading.Thread(target=run, daemon=True)
		stream_thread.start()


	def stop_recording_and_write():
		stop_event.set()
		# wait shortly for thread to finish
		if stream_thread is not None:
			stream_thread.join(timeout=1.0)
		mic.stop()
		# write wav
		now = datetime.now().strftime('%Y%m%d_%H%M%S')
		path = os.path.join('recordings', f'recording_{now}.wav')
		_write_wav(path, frames, rate, channels)
		print(f'Wrote {path} ({len(frames)} chunks)')


	def toggle():
		recording['on'] = not recording['on']
		if recording['on']:
			print('Recording started')
			start_recording()
		else:
			print('Recording stopped, writing WAV...')
			stop_recording_and_write()


	print('Press SPACE to toggle recording. Ctrl-C to quit.')
	listener = threading.Thread(target=_key_listener, args=(toggle,), daemon=True)
	listener.start()

	try:
		while True:
			time.sleep(0.1)
	except KeyboardInterrupt:
		if recording['on']:
			print('\nStopping active recording...')
			stop_recording_and_write()
		# Ensure PyAudio resources are released
		try:
			mic.close()
		except Exception:
			pass
		print('Exiting.')


if __name__ == '__main__':
	main()

