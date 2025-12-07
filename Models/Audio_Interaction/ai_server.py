from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import uvicorn
from car_assistant import transcribe_command, process_command, CAR_CONTEXT

app = FastAPI(title="AutoCar AI Server")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "online", "service": "AutoCar AI"}

@app.post("/process/text")
def process_text(request: TextRequest):
    """Process a text command directly."""
    response = process_command(request.text, CAR_CONTEXT)
    try:
        print(f"[AI_SERVER] /process/text input=\"{request.text}\" -> response=\"{response}\"")
    except Exception:
        pass
    return {"input": request.text, "response": response}

@app.post("/process/audio")
def process_audio(file: UploadFile = File(...)):
    """Upload an audio file (wav/ogg/mp3), transcribe it, and process the command."""
    temp_filename = f"temp_{file.filename}"
    
    try:
        # Save uploaded file temporarily
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Transcribe
        text = transcribe_command(temp_filename)
        
        # 2. Process
        response = process_command(text, CAR_CONTEXT)
        
        return {
            "transcription": text,
            "response": response
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    host = os.getenv("AI_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("AI_SERVER_PORT", "8010"))
    uvicorn.run(app, host=host, port=port)
