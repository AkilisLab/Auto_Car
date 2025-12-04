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
from spotify import play_music  # Import the player function

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
- If the user asks to PLAY MUSIC, output the command in this format: [[PLAY: Song Name]]
- If the user asks for an action (e.g., "go forward"), check if it's possible based on context, then confirm the action.
- If the user asks a question, answer concisely using the context.
- If the user requests emergency breaking, confirm the action.
- If the information is missing, say "I don't have that information."
- Keep answers short (2 to 3 sentences) suitable for text-to-speech.

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
    """Inject context and command into the LLM prompt."""
    
    # 1. Wake Word Check
    if not check_wake_word(command_text, "autodrive"):
        print("Wake word 'AutoDrive' not detected. Ignoring command.")
        return "(Ignored: Command did not contain 'AutoDrive')"

    # Remove wake word and common prefixes for cleaner processing
    # e.g. "Hey AutoDrive play music" -> "play music"
    # Regex explanation:
    # (hey|hi|okay|ok)? : Optional prefix
    # \s* : Optional whitespace
    # autodrive : The wake word
    # \W* : Optional non-word characters (punctuation)
    clean_command = re.sub(r'(hey|hi|okay|ok)?\s*autodrive\W*', '', command_text, flags=re.IGNORECASE).strip()
    
    if not clean_command:
        return "I'm listening. How can I help?"

    prompt = PromptTemplate.from_template(CAR_ASSISTANT_TEMPLATE).format(
        context=context_text,
        command=clean_command
    )
    
    print(f"Querying LLM ({llm_model})...")
    llm = Ollama(model=llm_model, temperature=0)
    response = llm(prompt)
    
    # Check for Function Calling (Music)
    match = re.search(r"\[\[PLAY: (.*?)\]\]", response)
    if match:
        song_query = match.group(1)
        print(f"Detected Music Intent: '{song_query}'")
        try:
            music_status = play_music(song_query)
            return f"{response}\n(System: {music_status})"
        except Exception as e:
            return f"{response}\n(System Error: Could not play music. {e})"

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
