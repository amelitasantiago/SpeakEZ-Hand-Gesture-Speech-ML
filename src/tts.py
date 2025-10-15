# C:\Users\ameli\speakez\src\tts.py
import threading
import queue
import time
import os
import tempfile
import traceback
import winsound

import pyttsx3
try:
    import pythoncom
except Exception:
    pythoncom = None


class _Speaker:
    def __init__(self, rate=180, volume=1.0, voice_name=None, max_queue=10):  # Cap queue
        self._q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._rate = rate
        self._volume = volume
        self._voice_name = voice_name
        self._thread = threading.Thread(target=self._worker, name="TTS-Worker", daemon=True)
        self._thread.start()

    def _worker(self):
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=0.1)
                if text is None:
                    self._q.task_done()
                    break
                text = str(text or "")  # Early coerce (handles int64/None)
                if not text.strip():
                    self._q.task_done()
                    continue
                text = text.strip()  # Safe now

                path = None
                try:
                    fd, path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)

                    if pythoncom:
                        try: pythoncom.CoInitialize()
                        except Exception: pass

                    eng = pyttsx3.init(driverName='sapi5')
                    eng.setProperty('rate', self._rate)
                    eng.setProperty('volume', self._volume)

                    print(f"[TTS] >> SYNTH(new eng) to {path}: {text!r}")
                    eng.save_to_file(text, path)
                    eng.runAndWait()
                    try: eng.stop()
                    except Exception: pass
                    del eng

                    # Async play: Offload to sub-thread to unblock queue
                    play_thread = threading.Thread(target=lambda: winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC), daemon=True)
                    play_thread.start()
                    play_thread.join(timeout=2.0)  # Wait short; drop if stalled
                    print(f"[TTS] << DONE {text!r}")
                except Exception as e:
                    print("[TTS] Worker error:", e)
                    traceback.print_exc()
                finally:
                    try:
                        if path and os.path.exists(path):
                            time.sleep(0.05)
                            os.remove(path)
                    except Exception:
                        pass
                    self._q.task_done()
            except queue.Empty:
                continue  # Poll loop for responsiveness

        print("[TTS] Worker exited.")

    def speak(self, text: str, rate: int = None, volume: float = None):
        if rate is not None: self._rate = rate
        if volume is not None: self._volume = volume
        self._q.put(text)

    def close(self, timeout: float = 1.0):
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=timeout)


_speaker = None
_lock = threading.RLock()

def _get_speaker():
    global _speaker
    with _lock:
        if _speaker is None:
            #_speaker = _Speaker()
            _speaker = _Speaker(max_queue=5)  # Limit backlog
        return _speaker

def tts_speak_live(text: str, rate: int = 180, volume: float = 1.0):
    sp = _get_speaker()
    sp.speak(text, rate=rate, volume=volume)

# --- BEGIN: add to src/tts.py ---
# If your file already defines tts_speak_live(text, rate=..., volume=...),
# add this alias so callers can do tts.speak("HELLO")
def speak(text: str, rate: int = 180, volume: float = 1.0):
    return tts_speak_live(text, rate=rate, volume=volume)
# --- END: add to src/tts.py ---
