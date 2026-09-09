"""clipgen — minimal long-form video → clip-moment pipeline.

Input a YouTube URL or a local video file → download + extract audio → transcribe
(WhisperX, optional diarization) → send the numbered transcript to Gemini → resolve the
returned moments into a ranked list of clips (real timestamps + text + per-dimension scores).
Driven entirely from a small web UI; there is no CLI workflow.
"""
