"""
Main script for simulated Pi client (webapp integration)
Handles UDP broadcasting, WebSocket connection, and delegates to mode handlers.
"""

# Imports for UDP, WebSocket, asyncio, etc.
import asyncio
import sys
import json
import time
import socket
import os
import base64
from contextlib import suppress

import cv2
import websockets

try:
    import sounddevice as sd
except Exception as exc:  # sounddevice may not be installed in some environments
    sd = None
    print(f"[WARN] sounddevice module unavailable: {exc}")

# Import mode handlers (to be implemented)
import sim_manual_mode
import sim_auto_mode
import sim_audio_mode

UDP_BROADCAST_PORT = 50010
UDP_LISTEN_PORT = 50011
UDP_BROADCAST_ADDR = '<broadcast>'
AI_SERVER_URL = os.getenv("AI_SERVER_URL", "http://192.168.1.255:8010")

# Device discovery

def broadcast_presence(device_id, info=None, listen_port=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    payload = json.dumps({
        "device_id": device_id,
        "info": info or "pi-sim",
        "action": "announce",
        "listen_port": listen_port or UDP_LISTEN_PORT,
    }).encode()
    sock.sendto(payload, (UDP_BROADCAST_ADDR, UDP_BROADCAST_PORT))
    sock.close()

def wait_for_connect(device_id, listen_sock, timeout=10, info="pi-sim", broadcast_interval=3.0):
    port = listen_sock.getsockname()[1]
    print(f"[{device_id}] Waiting for connect command from server on UDP port {port}...")
    deadline = time.time() + timeout
    next_broadcast = time.time()
    listen_sock.settimeout(1.0)
    try:
        while time.time() < deadline:
            try:
                data, addr = listen_sock.recvfrom(4096)
            except socket.timeout:
                data = None
            if data:
                try:
                    msg = json.loads(data.decode())
                except Exception:
                    continue
                if msg.get("action") == "connect" and msg.get("device_id") == device_id:
                    print(f"[{device_id}] Received connect command from {addr}")
                    return msg.get("ws_url", None)
            if time.time() >= next_broadcast:
                broadcast_presence(device_id, info=info, listen_port=port)
                next_broadcast = time.time() + broadcast_interval
    except Exception as exc:
        print(f"[{device_id}] Listener error while waiting for connect: {exc}")
    finally:
        listen_sock.close()
    print(f"[{device_id}] Timeout waiting for connect command.")
    return None

# Main WebSocket client
async def pi_client(uri: str, device_id: str = "pi-01", fps: float = 5.0, cam_index: int = 0):
    sim_audio_mode.set_ai_server_url(AI_SERVER_URL)
    sim_manual_mode.reset_state()
    sim_auto_mode.stop_autonomous_route("reset")
    sim_auto_mode.stop_grid_route()
    async with websockets.connect(uri) as ws:
        print(f"Connected to {uri} as {device_id}")
        handshake = {
            "role": "pi",
            "device_id": device_id,
            "action": "handshake",
            "payload": {"info": "pi-sim"},
        }
        await ws.send(json.dumps(handshake))

        async def receiver():
            try:
                async for message in ws:
                    try:
                        pkt = json.loads(message)
                    except Exception:
                        print("RECV (text):", message)
                        continue
                    act = pkt.get("action")
                    ptype = pkt.get("type")
                    payload = pkt.get("payload", {})
                    if isinstance(payload, dict):
                        ai_override = payload.get("ai_server_url")
                        if ai_override:
                            sim_audio_mode.set_ai_server_url(ai_override)
                    src = pkt.get("from") or pkt.get("device_id")
                    # Microphone open/close control
                    if act == "control" and ptype == "microphone_open":
                        print(f"[LOG] Received microphone_open event from {src} (payload: {payload})")
                        if not sd:
                            err = {
                                "role": "pi",
                                "device_id": device_id,
                                "action": "telemetry",
                                "type": "mic_transcript_error",
                                "payload": {"error": "sounddevice module not available"},
                                "ts": time.time(),
                            }
                            await ws.send(json.dumps(err))
                            continue
                        started = await sim_audio_mode.start_recording(sd)
                        if not started:
                            err = {
                                "role": "pi",
                                "device_id": device_id,
                                "action": "telemetry",
                                "type": "mic_transcript_error",
                                "payload": {"error": "failed_to_start_recording"},
                                "ts": time.time(),
                            }
                            await ws.send(json.dumps(err))
                        continue
                    elif act == "control" and ptype == "microphone_close":
                        print(f"[LOG] Received microphone_close event from {src} (payload: {payload})")
                        await sim_audio_mode.stop_and_send_recording(ws, device_id)
                        continue
                    # EMERGENCY STOP
                    if act == "control" and ptype == "emergency_stop":
                        sim_manual_mode.set_manual_flags(True, False, False)
                        sim_auto_mode.stop_autonomous_route("emergency_stop")
                        sim_auto_mode.stop_grid_route()
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "emergency_ack",
                            "payload": {
                                "status": "stopped",
                                "motors_disabled": True,
                                "emergency_timestamp": time.time(),
                                "device_id": device_id,
                                "route_stopped": True
                            },
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(ack))
                    elif act == "control" and ptype == "manual_drive":
                        spd = payload.get("speed")
                        ang = payload.get("angle")
                        success = sim_manual_mode.apply_manual_control(spd if spd is not None else 0.0, ang if ang is not None else 0.0)
                        state = sim_manual_mode.get_manual_state()
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "control_ack",
                            "payload": {
                                "applied_speed": state["current_speed"] if success else 0.0,
                                "applied_angle": state["current_angle"] if success else 0.0,
                                "motor_pwm": int(abs(state["current_speed"]) * 255) if success else 0,
                                "servo_position": 1500 + int(state["current_angle"] * 500) if success else 1500,
                                "emergency_active": state["emergency_active"],
                                "route_active": state["route_active"] or state["grid_route_active"],
                                "blocked": not success,
                                "last_command_ts": state["last_command_ts"],
                            },
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(ack))
                    elif act == "control" and ptype == "quick_command":
                        qc_text = payload.get("text", "")
                        await sim_audio_mode.handle_quick_command(ws, device_id, qc_text)
                    elif act == "control" and ptype == "auto_route_start":
                        destination = payload.get("destination", "Unknown")
                        settings = {
                            "route_type": payload.get("route_type", "fastest"),
                            "max_speed": payload.get("max_speed", 35),
                            "following_distance": payload.get("following_distance", "safe")
                        }
                        sim_auto_mode.start_autonomous_route(destination, settings)
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_ack",
                            "payload": {"status": "route_started", "destination": destination},
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(ack))
                    elif act == "control" and ptype == "auto_route_stop":
                        reason = payload.get("reason", "user_request")
                        sim_auto_mode.stop_autonomous_route(reason)
                        sim_auto_mode.stop_grid_route()
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_ack",
                            "payload": {"status": "route_stopped", "reason": reason},
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(ack))
                        route_status_update = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_status",
                            "payload": {
                                "status": "idle",
                                "destination": "",
                                "current_lat": 0,
                                "current_lng": 0,
                                "distance_remaining": 0,
                                "eta_minutes": 0,
                                "next_instruction": "",
                                "route_progress": 0,
                                "current_speed": 0,
                                "speed_limit": 35
                            },
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(route_status_update))
                    elif act == "control" and ptype == "execute_route":
                        wps = payload.get("waypoints") or []
                        goal_name = payload.get("goal_name", "Unknown")
                        if not wps:
                            sim_auto_mode.stop_grid_route()
                        else:
                            sim_auto_mode.start_grid_route(wps)
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_ack",
                            "payload": {
                                "status": "route_started" if wps else "route_invalid",
                                "destination": goal_name,
                                "waypoint_count": len(wps)
                            },
                            "ts": time.time()
                        }
                        await ws.send(json.dumps(ack))
                    elif act == "control" and ptype == "clear_emergency":
                        sim_manual_mode.clear_emergency()
                        ack = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "emergency_cleared_ack",
                            "payload": {"status": "normal_operation", "device_id": device_id},
                            "ts": time.time(),
                        }
                        await ws.send(json.dumps(ack))
                    else:
                        print("RECV PKT:", pkt)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print("Receiver stopped:", e)

        recv_task = asyncio.create_task(receiver())
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            print("Cannot open camera - continuing without video")
            cap = None
        try:
            interval = 1.0 / max(0.01, fps)
            while True:
                start = time.time()
                if cap:
                    ret, frame = cap.read()
                    if ret:
                        ok, enc = cv2.imencode(
                            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        )
                        if ok:
                            b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                            payload = {
                                "frame_b64": b64,
                                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                "encoding": "jpeg",
                            }
                            packet = {
                                "role": "pi",
                                "device_id": device_id,
                                "action": "telemetry",
                                "type": "camera_frame",
                                "payload": payload,
                                "ts": time.time(),
                            }
                            await ws.send(json.dumps(packet))
                auto_state = sim_auto_mode.get_auto_state()
                manual_state = sim_manual_mode.get_manual_state()
                status_payload = {
                    "speed": auto_state["simulated_speed"] if (auto_state["route_active"] or auto_state["grid_route_active"]) else manual_state["current_speed"],
                    "angle": manual_state["current_angle"],
                    "battery": 85.2,
                    "temperature": 42.1,
                    "connected": True,
                    "emergency_active": manual_state["emergency_active"],
                    "route_active": auto_state["route_active"] or auto_state["grid_route_active"],
                    "grid_route_active": auto_state["grid_route_active"],
                    "last_emergency": None,
                    "manual_last_command_ts": manual_state["last_command_ts"],
                    "route_seed": auto_state.get("seed"),
                }
                status_packet = {
                    "role": "pi",
                    "device_id": device_id,
                    "action": "telemetry",
                    "type": "status",
                    "payload": status_payload,
                    "ts": time.time(),
                }
                await ws.send(json.dumps(status_packet))
                if auto_state["route_active"]:
                    route_status = await sim_auto_mode.simulate_autonomous_navigation()
                    if route_status:
                        route_packet = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_status",
                            "payload": route_status,
                            "ts": time.time(),
                        }
                        await ws.send(json.dumps(route_packet))
                if auto_state["grid_route_active"]:
                    grid_status = await sim_auto_mode.simulate_grid_navigation()
                    if grid_status:
                        grid_packet = {
                            "role": "pi",
                            "device_id": device_id,
                            "action": "telemetry",
                            "type": "route_status",
                            "payload": grid_status,
                            "ts": time.time(),
                        }
                        await ws.send(json.dumps(grid_packet))
                        if grid_status["status"] == "arrived":
                            print("[ROUTE] Grid route completed.")
                elapsed = time.time() - start
                await asyncio.sleep(max(0, interval - elapsed))
        except websockets.exceptions.ConnectionClosedOK as e:
            print(f"WebSocket closed cleanly: code={e.code} reason={e.reason}")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"WebSocket closed by server: code={e.code} reason={e.reason}")
        except KeyboardInterrupt:
            print("\nShutting down pi simulator...")
        finally:
            if cap:
                cap.release()
            recv_task.cancel()
            with suppress(asyncio.CancelledError):
                await recv_task

if __name__ == "__main__":
    device_id = sys.argv[1] if len(sys.argv) > 1 else "pi-01"
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    try:
        while True:
            listen_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            listen_sock.bind(("", 0))
            listen_port = listen_sock.getsockname()[1]
            broadcast_presence(device_id, listen_port=listen_port)
            ws_url = wait_for_connect(device_id, listen_sock, info="pi-sim")
            if not ws_url:
                print(f"[{device_id}] No connect command received. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            asyncio.run(pi_client(ws_url, device_id=device_id, fps=fps))
            print(f"[{device_id}] WebSocket session ended. Returning to standby.")
    except KeyboardInterrupt:
        print("\nShutting down pi simulator...")
