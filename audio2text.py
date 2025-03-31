import whisper


def transcribe_audio(whisper_model, audio_path):
    """
    Transcribes the audio file using Whisper.
    Uses task="translate" to convert non-English speech into English.
    """
    result = whisper_model.transcribe(audio_path, task="translate", language="en")
    return result["text"]