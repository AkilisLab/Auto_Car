"""Audio mode handler for simulated Pi client."""

from __future__ import annotations

import json
import tempfile
import time
import wave
from typing import Optional

import numpy as np
import requests

AI_SERVER_URL: Optional[str] = None  # Set this from main script or config

SAMPLERATE = 16000
_recording_stream = None
_audio_buffer: list[np.ndarray] = []


def set_ai_server_url(url: Optional[str]) -> None:
    global AI_SERVER_URL
    AI_SERVER_URL = url.rstrip("/") if url else None


def _audio_callback(indata, frames, time_info, status):  # pragma: no cover - sounddevice callback
    _audio_buffer.append(indata.copy())


def is_recording() -> bool:
    return _recording_stream is not None


async def start_recording(sd_module) -> bool:
    global _recording_stream, _audio_buffer
    if _recording_stream:
        print("[MIC] Recording already active. Ignoring duplicate start.")
        return False
    try:
        _audio_buffer = []
        _recording_stream = sd_module.InputStream(
            samplerate=SAMPLERATE,
            channels=1,
            dtype="int16",
            callback=_audio_callback,
        )
        _recording_stream.start()
        print("[MIC] Microphone recording started (Siri-style)")
        return True
    except Exception as exc:
        _recording_stream = None
        print(f"[MIC][ERROR] Failed to start recording: {exc}")
        return False


async def stop_and_send_recording(ws, device_id: str) -> None:
    global _recording_stream
    if _recording_stream:
        _recording_stream.stop()
        _recording_stream.close()
        _recording_stream = None
        print("[MIC] Microphone recording stopped")
    if not _audio_buffer:
        print("[MIC] No audio recorded.")
        return
    audio_data = np.concatenate(_audio_buffer, axis=0)
    _audio_buffer.clear()
    if audio_data.dtype != np.int16:
        audio_data = audio_data.astype(np.int16)
    if not AI_SERVER_URL:
        print("[MIC][ERROR] AI server URL not configured")
        return
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="sim_audio_", suffix=".wav", delete=False) as tmpfile:
            tmp_path = tmpfile.name
        with wave.open(tmp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLERATE)
            wav_file.writeframes(audio_data.tobytes())
        with open(tmp_path, "rb") as wav_bytes:
            files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
            resp = requests.post(f"{AI_SERVER_URL}/process/audio", files=files, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        msg = {
            "role": "pi",
            "device_id": device_id,
            "action": "telemetry",
            "type": "mic_transcript",
            "payload": data,
            "ts": time.time(),
        }
        await ws.send(json.dumps(msg))
        print(f"[MIC] Transcription sent: {data}")
    except Exception as exc:
        error_msg = {
            "role": "pi",
            "device_id": device_id,
            "action": "telemetry",
            "type": "mic_transcript_error",
            "payload": {"error": str(exc)},
            "ts": time.time(),
        }
        print(f"[MIC][ERROR] {exc}")
        try:
            await ws.send(json.dumps(error_msg))
        except Exception:
            pass
    finally:
        if tmp_path:
            import os

            if os.path.exists(tmp_path):
                os.remove(tmp_path)


async def handle_quick_command(ws, device_id: str, qc_text: str, spotify=None) -> None:
    ack = {
        "role": "pi",
        "device_id": device_id,
        "action": "telemetry",
        "type": "quick_command_ack",
        "payload": {"received_text": qc_text},
        "ts": time.time(),
    }
    await ws.send(json.dumps(ack))
    result_payload = {"input": qc_text, "response": None, "error": None}
    if not AI_SERVER_URL:
        result_payload["error"] = "AI server URL not configured"
    else:
        try:
            url = f"{AI_SERVER_URL}/process/text"
            resp = requests.post(url, json={"text": qc_text}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            result_payload["response"] = data.get("response")
            if data.get("input"):
                result_payload["input"] = data["input"]
            print(f"[CLIENT] AI response: {result_payload['response']}")
        except Exception as exc:
            result_payload["error"] = str(exc)
            print(f"[CLIENT][ERROR] AI server call failed: {exc}")
    result_msg = {
        "role": "pi",
        "device_id": device_id,
        "action": "telemetry",
        "type": "command_result",
        "payload": result_payload,
        "ts": time.time(),
    }
    await ws.send(json.dumps(result_msg))
    music_query = None
    lowered = qc_text.lower()
    if lowered.startswith("play "):
        music_query = qc_text[5:].strip()
    elif lowered.startswith("spotify "):
        music_query = qc_text[8:].strip()
    elif isinstance(result_payload.get("response"), str):
        resp_txt = result_payload["response"].lower()
        if resp_txt.startswith("play_music:"):
            music_query = result_payload["response"].split(":", 1)[1].strip()
    if music_query and spotify:
        try:
            print(f"[CLIENT][SPOTIFY] Triggering playback for query: {music_query}")
            play_result = spotify.play_music(music_query)
            music_msg = {
                "role": "pi",
                "device_id": device_id,
                "action": "telemetry",
                "type": "music_play",
                "payload": {"query": music_query, "status": play_result},
                "ts": time.time(),
            }
            await ws.send(json.dumps(music_msg))
        except Exception as exc:
            print(f"[CLIENT][SPOTIFY][ERROR] {exc}")
            err_msg = {
                "role": "pi",
                "device_id": device_id,
                "action": "telemetry",
                "type": "music_play",
                "payload": {"query": music_query, "error": str(exc)},
                "ts": time.time(),
            }
            await ws.send(json.dumps(err_msg))
