#!/usr/bin/env python3
"""Pi WebSocket client (manual mode) controlled by frontend.

Replaces local keyboard control with joystick/button events coming from the
web UI via the FastAPI WebSocket relay.

Protocol (compatible with Auto_Car_Webapp/backend/server.py):
    Handshake (on connect):
        {"role":"pi","device_id":"pi-01","action":"handshake","payload":{"capabilities":["manual"],"version":"1.0"}}
    Frontend -> Pi control packets (server-routed):
        manual drive:
            {"action":"control","type":"manual_drive","payload":{"speed":-1..1,"angle":-1..1}}
        emergency:
            {"action":"control","type":"emergency_stop"} | {"type":"clear_emergency"}

Pi telemetry responses:
    control_ack, emergency_ack, emergency_cleared_ack (mirroring simulator in backend/client.py)

Motor mapping:
    speed -> PWM (forward-only clamp for v1); angle -> differential mix.

CLI Usage (mimics backend/client.py):
    python3 pi_ws_client.py ws://<server-ip>:8000/ws [device_id] [loop_hz] [base_pwm] [watchdog_s]

Environment variable fallbacks still supported (WS_URL, DEVICE_ID, LOOP_HZ, BASE_PWM, WATCHDOG_S).

Dependencies:
    websockets (async WebSocket client). Install: pip install websockets
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
import base64

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # allow running without OpenCV, camera streaming will be disabled

try:
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover
    print("Missing dependency: websockets. Install with: pip install websockets", file=sys.stderr)
    raise


# Hardware interfaces from this repo
try:
    from devices.raspbot import Raspbot
except Exception as e:  # pragma: no cover - allow import error to surface clearly
    print("Failed to import devices.raspbot.Raspbot. Ensure your PYTHONPATH includes the repo root.")
    raise


def clamp(v, a, b):
    return a if v < a else b if v > b else v


@dataclass
class ControlState:
    speed: float = 0.0  # -1..1 (reverse..forward) – currently clamped to 0..1
    angle: float = 0.0  # -1..1 (left..right)
    last_update_ts: float = 0.0
    emergency: bool = False


class MotorController:
    """Maps speed/angle into per-wheel PWM and commands the Raspbot."""

    def __init__(self, base_pwm: int = 120, steer_gain: float = 1.0):
        self.bot = Raspbot()
        self.base_pwm = int(clamp(base_pwm, 0, 255))
        self.steer_gain = float(steer_gain)

    def apply(self, speed: float, angle: float) -> None:
        """Apply joystick speed/angle.

        speed: -1..1 (negative = reverse)
        angle: -1..1 (negative = turn left)

        Direction semantics:
          Forward  -> direction flag 0
          Reverse  -> direction flag 1 (assumed; adjust if hardware differs)

        Steering while reversing is intuitively inverted (so pushing left
        still causes a left heading change) by flipping differential sign.
        """
        sp = clamp(speed, -1.0, 1.0)
        ang = clamp(angle, -1.0, 1.0)

        forward = abs(sp)
        base = int(self.base_pwm * forward)

        # Differential component. For reverse we invert to keep intuitive steering.
        diff = int(base * self.steer_gain * ang)
        if sp < 0:
            diff = -diff

        left_pwm = clamp(base + diff, 0, 255)
        right_pwm = clamp(base - diff, 0, 255)
        direction_flag = 0 if sp >= 0 else 1

        try:
            self.bot.Ctrl_Car(0, direction_flag, left_pwm)
            self.bot.Ctrl_Car(1, direction_flag, left_pwm)
            self.bot.Ctrl_Car(2, direction_flag, right_pwm)
            self.bot.Ctrl_Car(3, direction_flag, right_pwm)
        except Exception:
            pass

    def stop(self) -> None:
        try:
            self.bot.Ctrl_Car(0, 0, 0)
            self.bot.Ctrl_Car(1, 0, 0)
            self.bot.Ctrl_Car(2, 0, 0)
            self.bot.Ctrl_Car(3, 0, 0)
        except Exception:
            pass


class PiManualClient:
    def __init__(self, ws_url: str, device_id: str = "pi-01", loop_hz: float = 30.0, watchdog_s: float = 0.7, base_pwm: int = 120,
                 cam_fps: float = 5.0, cam_index: int = 0, steer_gain: float = 1.0, hold_timeout_s: float = 10.0):
        self.ws_url = ws_url
        self.device_id = device_id
        self.loop_hz = loop_hz
        self.watchdog_s = watchdog_s
        self.control = ControlState()
        self.motors = MotorController(base_pwm=base_pwm, steer_gain=steer_gain)
        self.ws = None  # type: ignore
        self._stop_evt = asyncio.Event()
        self.cam_fps = cam_fps
        self.cam_index = cam_index
        self._camera = None
        self.hold_timeout_s = hold_timeout_s

    # --------------- connection & protocol helpers ---------------
    async def _handshake(self, ws) -> None:
        payload = {"role": "pi", "device_id": self.device_id, "action": "handshake", "payload": {"capabilities": ["manual", "camera", "status"], "version": "1.0"}}
        await ws.send(json.dumps(payload))

    async def _send(self, ws, obj: dict) -> None:
        try:
            await ws.send(json.dumps(obj))
        except Exception:
            pass

    async def _send_ack(self, ws, ack_type: str, payload: dict) -> None:
        msg = {"from": self.device_id, "action": "telemetry", "type": ack_type, "payload": payload, "ts": time.time()}
        await self._send(ws, msg)

    # --------------- control loops ---------------
    async def recv_loop(self, ws) -> None:
        while not self._stop_evt.is_set():
            try:
                raw = await ws.recv()
            except Exception:
                break  # connection closed -> outer reconnect

            try:
                pkt = json.loads(raw)
            except Exception:
                # Ignore malformed messages
                continue

            act = pkt.get("action")
            ptype = pkt.get("type")
            payload = pkt.get("payload", {})

            # Frontend control -> apply
            if act == "control":
                if ptype == "manual_drive":
                    # Frontend supplies speed in -1..1 (or -100..100 scaled); normalize if >1.
                    raw_sp = float(payload.get("speed", 0.0))
                    # Accept values in percentage form (-100..100) by scaling down.
                    sp = raw_sp / 100.0 if abs(raw_sp) > 1.0 else raw_sp
                    ang = float(payload.get("angle", 0.0))
                    # update state; watchdog will stop if stale
                    self.control.speed = sp
                    self.control.angle = ang
                    self.control.last_update_ts = time.time()
                    await self._send_ack(ws, "control_ack", {"applied": True, "speed": sp, "angle": ang, "emergency_active": self.control.emergency})
                    continue

                if ptype == "emergency_stop":
                    self.control.emergency = True
                    self.motors.stop()
                    await self._send_ack(ws, "emergency_ack", {"stopped": True})
                    continue

                if ptype == "clear_emergency":
                    self.control.emergency = False
                    await self._send_ack(ws, "emergency_cleared_ack", {"cleared": True})
                    continue

            # Other messages are ignored in manual-only client

    async def control_loop(self) -> None:
        interval = 1.0 / max(1.0, float(self.loop_hz))
        while not self._stop_evt.is_set():
            now = time.time()
            try:
                if self.control.emergency:
                    self.motors.stop()
                else:
                    stale = now - self.control.last_update_ts
                    # If command is stale and speed is near zero, stop. Otherwise, keep holding last command
                    # up to hold_timeout_s to avoid unintended runaway.
                    if stale > self.hold_timeout_s:
                        self.motors.stop()
                    else:
                        if stale > self.watchdog_s and abs(self.control.speed) < 1e-3:
                            self.motors.stop()
                        else:
                            self.motors.apply(self.control.speed, self.control.angle)
            except Exception:
                # Never crash the loop; attempt again next tick
                pass
            await asyncio.sleep(interval)

    async def status_loop(self, ws) -> None:
        """Periodically send status telemetry similar to simulator client."""
        while not self._stop_evt.is_set():
            payload = {
                "speed": float(self.control.speed),
                "angle": float(self.control.angle),
                "emergency_active": bool(self.control.emergency),
                "connected": True,
            }
            pkt = {"role": "pi", "device_id": self.device_id, "action": "telemetry", "type": "status", "payload": payload, "ts": time.time()}
            await self._send(ws, pkt)
            await asyncio.sleep(1.0)

    async def camera_loop(self, ws) -> None:
        """Capture frames and send as base64 JPEG at configured FPS.

        Uses devices.camera_node.Camera if available; falls back to cv2.VideoCapture.
        """
        interval = 1.0 / max(0.1, float(self.cam_fps))
        Camera = None
        try:
            from devices.camera_node import Camera as _Cam
            Camera = _Cam
        except Exception:
            Camera = None

        cap = None
        started = False
        try:
            if Camera is not None:
                try:
                    cam = Camera(index=self.cam_index, width=640, height=480, fps=int(self.cam_fps))
                    # Prefer background thread reader
                    cam.start()
                    self._camera = cam
                    started = True
                except Exception:
                    self._camera = None
                    started = False
            if not started and cv2 is not None:
                cap = cv2.VideoCapture(self.cam_index)
                if not cap or not cap.isOpened():
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    cap = None

            while not self._stop_evt.is_set():
                frame = None
                if self._camera is not None:
                    try:
                        # prefer latest frame; fallback to blocking read
                        ret, fr = (False, None)
                        try:
                            ret, fr = self._camera.get_frame()
                        except Exception:
                            ret, fr = self._camera.read(wait=True)
                        if ret and fr is not None:
                            frame = fr
                    except Exception:
                        frame = None
                elif cap is not None and cv2 is not None:
                    try:
                        ret, fr = cap.read()
                        if ret and fr is not None:
                            frame = fr
                    except Exception:
                        frame = None

                if frame is not None and cv2 is not None:
                    try:
                        ok, enc = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            b64 = base64.b64encode(enc.tobytes()).decode('ascii')
                            payload = {
                                "frame_b64": b64,
                                "width": int(frame.shape[1]),
                                "height": int(frame.shape[0]),
                                "encoding": "jpeg",
                            }
                            pkt = {"role": "pi", "device_id": self.device_id, "action": "telemetry", "type": "camera_frame", "payload": payload, "ts": time.time()}
                            await self._send(ws, pkt)
                    except Exception:
                        pass

                await asyncio.sleep(interval)
        finally:
            # cleanup
            if self._camera is not None:
                try:
                    try:
                        self._camera.stop()
                    except Exception:
                        pass
                    self._camera.close()
                except Exception:
                    pass
                self._camera = None
            if cap is not None and cv2 is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    # --------------- lifecycle ---------------
    async def run_once(self) -> None:
        async with websockets.connect(self.ws_url, max_size=4 * 1024 * 1024) as ws:  # generous default
            self.ws = ws
            await self._handshake(ws)
            # run recv, control, camera, status loops concurrently
            tasks = {
                asyncio.create_task(self.recv_loop(ws)),
                asyncio.create_task(self.control_loop()),
            }
            # camera and status loops are optional; they should not crash client if camera missing
            tasks.add(asyncio.create_task(self.status_loop(ws)))
            tasks.add(asyncio.create_task(self.camera_loop(ws)))

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # cancel the other task if one completes
            for t in pending:
                t.cancel()
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception:
                pass

    async def run_forever(self) -> None:
        backoff = 1.0
        while not self._stop_evt.is_set():
            try:
                await self.run_once()
                backoff = 1.0  # reset after a successful session
            except Exception:
                # connection error; exponential backoff up to 15s
                await asyncio.sleep(backoff)
                backoff = min(15.0, backoff * 2.0)

    def stop(self) -> None:
        self._stop_evt.set()
        # ensure motors are stopped
        try:
            self.motors.stop()
        except Exception:
            pass


def _install_signal_handlers(client: PiManualClient):
    def handle(sig, frame):  # noqa: ANN001
        client.stop()
    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)


async def _amain(ws_url: str, device_id: str, loop_hz: float, base_pwm: int, watchdog_s: float, cam_fps: float, cam_index: int,
                 steer_gain: float, hold_timeout_s: float) -> None:
    client = PiManualClient(ws_url=ws_url, device_id=device_id, loop_hz=loop_hz, watchdog_s=watchdog_s, base_pwm=base_pwm,
                            cam_fps=cam_fps, cam_index=cam_index, steer_gain=steer_gain, hold_timeout_s=hold_timeout_s)
    _install_signal_handlers(client)
    print(f"[pi-client] connecting to {ws_url} as {device_id} (base_pwm={base_pwm}, loop_hz={loop_hz}, watchdog={watchdog_s}s, cam_fps={cam_fps}, cam_index={cam_index})")
    try:
        await client.run_forever()
    finally:
        client.stop()


def main():
    # Positional CLI arguments mimic backend/client.py style.
    #   pi_ws_client.py ws://host:8000/ws [device_id] [loop_hz] [base_pwm] [watchdog_s]
    argv = sys.argv
    if len(argv) >= 2 and argv[1].startswith("ws://") or (len(argv) >= 2 and argv[1].startswith("wss://")):
        ws_url = argv[1]
    else:
        ws_url = os.environ.get("WS_URL", "ws://localhost:8000/ws")
    device_id = argv[2] if len(argv) > 2 else os.environ.get("DEVICE_ID", "pi-01")
    try:
        # If only three args are provided (ws_url, device_id, X), mimic backend/client.py and treat X as camera FPS.
        if len(argv) == 4:
            loop_hz = float(os.environ.get("LOOP_HZ", "30"))
            cam_fps = float(argv[3])
        else:
            loop_hz = float(argv[3]) if len(argv) > 3 else float(os.environ.get("LOOP_HZ", "30"))
            cam_fps = float(os.environ.get("FPS", "5.0"))
    except Exception:
        loop_hz = 30.0
        cam_fps = float(os.environ.get("FPS", "5.0"))
    try:
        base_pwm = int(argv[4]) if len(argv) > 4 else int(os.environ.get("BASE_PWM", "120"))
    except Exception:
        base_pwm = 120
    try:
        watchdog_s = float(argv[5]) if len(argv) > 5 else float(os.environ.get("WATCHDOG_S", "0.7"))
    except Exception:
        watchdog_s = 0.7
    try:
        # If six args provided and we already used argv[3] for loop_hz, argv[6] may be cam_fps
        if len(argv) > 6:
            cam_fps = float(argv[6])
    except Exception:
        pass
    try:
        cam_index = int(argv[7]) if len(argv) > 7 else int(os.environ.get("CAM_INDEX", "0"))
    except Exception:
        cam_index = 0
    try:
        steer_gain = float(argv[8]) if len(argv) > 8 else float(os.environ.get("STEER_GAIN", "1.0"))
    except Exception:
        steer_gain = 1.0
    try:
        hold_timeout_s = float(argv[9]) if len(argv) > 9 else float(os.environ.get("HOLD_TIMEOUT_S", "10.0"))
    except Exception:
        hold_timeout_s = 10.0

    # Help output if explicitly requested
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(
            "Usage:\n"
            "  python3 pi_ws_client.py ws://<server-ip>:8000/ws [device_id] [fps]\n"
            "  python3 pi_ws_client.py ws://<server-ip>:8000/ws [device_id] [loop_hz] [base_pwm] [watchdog_s] [fps] [cam_index] [steer_gain] [hold_timeout_s]\n"
            "Env fallbacks: WS_URL, DEVICE_ID, LOOP_HZ, BASE_PWM, WATCHDOG_S, FPS, CAM_INDEX, STEER_GAIN, HOLD_TIMEOUT_S"
        )
        return

    try:
        asyncio.run(_amain(ws_url, device_id, loop_hz, base_pwm, watchdog_s, cam_fps, cam_index, steer_gain, hold_timeout_s))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
