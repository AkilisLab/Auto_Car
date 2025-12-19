"""Car Assistant using Prompt Stuffing.

Adapts the 'Alternative RAG' approach from qa_revised.py for an Auto Car context.
In the original script, the Audio was the Context (Knowledge) and the Question was text.
Here, we invert it:
- Context: Static Car Manual / Status (Text)
- Question: User's Voice Command (Audio -> Text)

Usage:
    python car_assistant.py --audio command.ogg
"""
import argparse
import sys
import whisper
import re
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# ------------------ Mock Car Knowledge ------------------
# In a real system, this could be loaded from a PDF manual or dynamic vehicle status API.
CAR_CONTEXT = """
Vehicle: AutoCar Pi (2025)
System Status:
- Battery Level: 85%
- Current Location: 123 Main St, Springfield

Owner's Manual Quick Reference:
1. Climate Control: Say "Set temperature to X degrees" to adjust.
2. Navigation: Say "Navigate to [Destination]" to start route.
3. Media: Supports "Play [Song/Artist]" via Spotify integration.
4. Safety: Automatic Emergency Braking is allowed.
5. Maintenance: Next service due in 3,000 miles.
6. Voice Assistant: Can answer questions about vehicle status and manual.
"""

# ------------------ Prompt Template ------------------
# Adapted to act as a Car Assistant
CAR_ASSISTANT_TEMPLATE = """You are an intelligent in-car voice assistant for the AutoCar.
Use the provided 'Car Context' (System Status and Manual) to answer the user's voice command or question.

Guidelines:
- Keep answers short (1 sentence if possible) suitable for text-to-speech.
- Do NOT include greetings, quotes, or multi-paragraph explanations.
- Do NOT ask follow-up questions for navigation. If the user specifies a destination name (e.g., "home"), proceed.
- Do NOT emit music intents unless the user explicitly asked to play music.
- For navigation requests ("navigate", "go to", "drive to", "take me to"), ALWAYS use auto navigation intents (auto_route_start/auto_route_stop) — never timed_drive.

IMPORTANT OUTPUT FORMAT (for downstream control parsing):
ALWAYS end your response with exactly ONE machine-readable intent line in one of these forms:
    - [[INTENT: play_music; query="<song or artist query>"]]
    - [[INTENT: timed_drive; direction=forward|backward|left|right; duration_s=<number>]]
    - [[INTENT: auto_route_start; destination="<place>"; route_type=fastest|eco|safe; max_speed=<number>]]
    - [[INTENT: auto_route_stop; reason="user_request"]]
    - [[INTENT: stop]]
    - [[INTENT: emergency_stop]]
    - [[INTENT: none]]
Rules for the intent line:
- It must be the final line of your output.
- Use only one intent line.
- Use duration_s in seconds when applicable.
- If a parameter is not specified by the user, choose a reasonable default (route_type=fastest, max_speed=35).
- If the destination is a name (not coordinates), put the name into destination exactly (e.g., destination="home").

Examples (follow these formats exactly):
User: Navigate home
Assistant: Starting navigation.
[[INTENT: auto_route_start; destination="home"; route_type=fastest; max_speed=35]]

User: Stop navigation
Assistant: Stopping navigation.
[[INTENT: auto_route_stop; reason="user_request"]]

User: Go forward for 3 seconds
Assistant: Moving forward.
[[INTENT: timed_drive; direction=forward; duration_s=3]]

Car Context:
{context}

User Voice Command:
{command}

Assistant Response:"""


# ------------------ Transcription ------------------
def transcribe_command(audio_path: str, model_name: str = "base") -> str:
    """Transcribe a short voice command using Whisper."""
    print(f"Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"Transcribing command from '{audio_path}'...")
    result = model.transcribe(audio_path, fp16=False)
    text = result.get("text", "").strip()
    return text


# ------------------ Wake Word Logic ------------------
def check_wake_word(text: str, wake_word: str = "autocar") -> bool:
    """Check if the command contains the wake word (case-insensitive)."""
    text_lower = text.lower()
    wake_word_lower = wake_word.lower()
    
    # Allow "Hey AutoCar", "Hi AutoCar", "Okay AutoCar", or just "AutoCar"
    # We check if the wake word appears in the text.
    if wake_word_lower in text_lower:
        return True
    return False


# ------------------ Prompt Stuffing Logic ------------------
def process_command(command_text: str, context_text: str, llm_model: str = "llama2") -> str:
    """Inject context and command into the LLM prompt.

    Note: This function does NOT execute actions (e.g., Spotify playback).
    If the model decides music should be played, it should emit a tag like
    `[[PLAY: Song Name]]` which downstream components (client/backend) can
    interpret and execute.
    """

    # 1) Wake word check (do not change: AutoDrive)
    if not check_wake_word(command_text, "autodrive"):
        print("Wake word 'AutoDrive' not detected. Ignoring command.")
        return "(Ignored: Command did not contain 'AutoDrive')"

    # 2) Strip wake word + common prefixes
    clean_command = re.sub(
        r"(hey|hi|okay|ok)?\s*autodrive\W*",
        "",
        command_text or "",
        flags=re.IGNORECASE,
    ).strip()

    if not clean_command:
        return "I'm listening. How can I help?"

    prompt = PromptTemplate.from_template(CAR_ASSISTANT_TEMPLATE).format(
        context=context_text,
        command=clean_command,
    )

    print(f"Querying LLM ({llm_model})...")
    llm = Ollama(model=llm_model, temperature=0)
    response = llm(prompt)

    # Do not execute any side-effects here; return the model output verbatim.
    return response


# ------------------ Main ------------------
def main():
    parser = argparse.ArgumentParser(description="Auto Car Voice Assistant")
    parser.add_argument("--audio", help="Path to audio file containing voice command")
    parser.add_argument("--text", help="Text command (skip transcription for testing)")
    parser.add_argument("--model", default="llama2", help="Ollama model name")
    args = parser.parse_args()

    if not args.audio and not args.text:
        print("Error: Please provide --audio [file] or --text [string]")
        sys.exit(1)

    # 1. Get Command (Audio -> Text OR Direct Text)
    if args.audio:
        try:
            command_text = transcribe_command(args.audio)
        except Exception as e:
            print(f"Transcription failed: {e}")
            sys.exit(1)
    else:
        command_text = args.text

    print(f"\nUser Command: \"{command_text}\"\n")

    # 2. Process with Context (Prompt Stuffing)
    response = process_command(command_text, CAR_CONTEXT, args.model)

    print("-" * 30)
    print("AutoCar Assistant:", response)
    print("-" * 30)


if __name__ == "__main__":
    main()
