# Audio Interaction — AI Server

This folder contains the AI server used by the Auto Car stack to process text and audio commands. It wraps logic from `car_assistant.py`.

## Endpoints
- `GET /` — health check
- `POST /process/text` — `{ "text": "..." }` → `{ "input": "...", "response": "..." }`
- `POST /process/audio` — upload `file` (`wav/ogg/mp3`) → `{ "transcription": "...", "response": "..." }`

## Run (port 8010)
```
cd /home/akilis/Documents/GitHub/Auto_Car/Models/Audio_Interaction
python3 -m uvicorn ai_server:app --host 0.0.0.0 --port 8010 --reload
uvicorn ai_server:app --host 0.0.0.0 --port 8010
```

Quick test:
```
curl http://127.0.0.1:8010/
curl -X POST http://127.0.0.1:8010/process/text -H "Content-Type: application/json" -d '{"text":"Hey AutoDrive, navigate home"}'
```

## Integration
- The Pi simulator (`Auto_Car_Webapp/backend/client.py`) sends Quick Commands to `/process/text` and relays the response back to the backend/frontend.
- For audio capture (future), the client can upload chunks/files to `/process/audio`.

## Requirements
Install with a virtual environment (recommended):
```
python -m venv audio_env
source audio_env/bin/activate
pip install -r requirements.txt
```

## Logging
`/process/text` prints a concise line for observability:
```
[AI_SERVER] /process/text input="..." -> response="..."
```

---

Legacy scripts from earlier RAG experiments remain in this folder (e.g., `rag_over_whisper.py`). They are not required for the webapp quick commands path, but can be used independently.
