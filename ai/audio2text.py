import whisper


def transcribe_audio(whisper_model, audio_path, language_code="es"):
    """
    Transcribes the audio file using Whisper.
    Uses task="translate" to convert non-English speech into English.
    """
    result = whisper_model.transcribe(audio_path, task="translate", language=language_code)
    return result["text"]