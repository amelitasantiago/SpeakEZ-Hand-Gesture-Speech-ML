# tts_gtts.py
import tempfile, os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

def tts_speak_live(text, rate=1.0):
    # gTTS is always 1x speed; rate ignored or implement via time-stretch
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        gTTS(text=text, lang='en').save(tmp)
        snd = AudioSegment.from_file(tmp, format="mp3")
        play(snd)  # blocks until done; releases device
    finally:
        try: os.remove(tmp)
        except OSError: pass



# tts.py
import threading
import pyttsx3

_engine = None
_lock = threading.RLock()

def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            eng = pyttsx3.init()  # SAPI5 on Windows
            # Optional voice tweaking:
            # voices = eng.getProperty('voices'); eng.setProperty('voice', voices[0].id)
            _engine = eng
        return _engine

def tts_speak_live(text: str, rate: int = 180, volume: float = 1.0):
    """Speak text reliably even when called repeatedly from a video loop."""
    if not text:
        return
    eng = _get_engine()
    with _lock:
        # ensure fresh settings every time
        eng.setProperty('rate', rate)
        eng.setProperty('volume', volume)
        # stop any buffered/ongoing utterance to avoid queue buildup
        try:
            eng.stop()
        except Exception:
            pass
        eng.say(text)
        # runAndWait must complete; it pumps the audio loop
        eng.runAndWait()
