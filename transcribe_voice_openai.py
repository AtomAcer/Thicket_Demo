# package imports
import tempfile
import base64
import streamlit as st
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]


def record_and_transcribe(client, audio_bytes):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        # Write the audio bytes to the temporary file
        temp_audio_file.write(audio_bytes)
        temp_audio_file.flush()  # Ensure the data is written
        temp_audio_file_path = temp_audio_file.name  # Get the file path

    # Pass the temporary file path to the model for transcription
    with open(temp_audio_file_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="text",
            language="en"
        )

    # Clean up the temporary file (optional, you might want to ensure deletion)
    try:
        os.remove(temp_audio_file_path)
    except OSError as e:
        print(f"Error deleting temporary file: {e}")

    return transcript




def create_output_speech(client, response_text, voice="alloy"):
    """
    Create a speech output from text using TTS model and save as WAV file.

    Args:
        client: API client for audio processing
        response_text (str): Text to be converted to speech
        voice (str): Voice style for the speech, default is "alloy"

    Returns:
        None
    """
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=response_text,
    ) as response:
        response.stream_to_file("speech.wav")
    return

def convert_audio_to_base64(audio_file_path):
    """
    Convert an audio file to base64 encoding.

    Args:
        audio_file_path (str): Path to the audio file

    Returns:
        str: Base64 encoded string of the audio file
    """
    with open(audio_file_path, 'rb') as f:
        audio_encoded = base64.b64encode(f.read()).decode('utf-8')
    return audio_encoded