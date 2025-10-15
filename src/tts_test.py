# tools/tts_test.py
from tts import tts_speak_live, tts_to_mp3_bytes
tts_speak_live("Hello from SpeakEZ. This is a quick test.")
mp3 = tts_to_mp3_bytes("This MP3 should be downloadable in the UI.")
open("test.mp3","wb").write(mp3)
print("Done.")
