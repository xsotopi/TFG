import whisper


def transcribe_audio(whisper_model, audio_path, language_code):
    """
    Transcribes the audio file using Whisper.
    Uses task="translate" to convert non-English speech into English.
    Returns:
        str: The transcribed and translated text in English.
    """
    result = whisper_model.transcribe(audio_path, task="translate", language=language_code)

    return result["text"]