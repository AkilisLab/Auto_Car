# Auto_Car
An auto car project, @NTUST.

## Simulator Quick Start
- Activate the local venv: `source car_env/bin/activate` from `Car_main/`.
- Make sure the control server/webapp that listens for Pi announcements is running.
- Run the simulator entry point: `python sim_client_main.py [device_id] [fps]` (defaults `pi-01` and `5`).
- The script will broadcast over UDP until the server replies with a WebSocket URL, then stream camera frames, telemetry, and handle manual/auto/audio commands through `sim_manual_mode.py`, `sim_auto_mode.py`, and `sim_audio_mode.py`.
- Optional: set `AI_SERVER_URL` before launching if you need to forward audio transcripts to a different service.

## Python Requirements
- `numpy`
- `opencv-contrib-python` (provides `cv2` and `cv2.aruco`)
- `websockets`
- `requests`
- `sounddevice`
- `scipy`
- `pyaudio`
- `pydub`
- `simpleaudio`
- `Pillow` (imported as `PIL`)
- `Adafruit-SSD1306`
- `smbus` (or `smbus2` on non-Raspberry Pi hosts)
- `spotipy` (required by spotify.py helper when Spotify integration is enabled)

### Notes
- `pydub` expects `ffmpeg` to be installed on the host OS for audio playback.
- Hardware-oriented packages (`Adafruit-SSD1306`, `smbus`) are only required when driving the physical OLED display and I2C peripherals.