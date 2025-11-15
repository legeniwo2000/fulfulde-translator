import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("🎙️ Say something...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
    print("✅ Got your audio! Now converting...")

try:
    text = recognizer.recognize_google(audio)
    print("🗣️ You said:", text)
except Exception as e:
    print("❌ Error:", e)
