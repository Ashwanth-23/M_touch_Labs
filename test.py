

from elevenlabs.client import ElevenLabs

# Create client
client = ElevenLabs(api_key="sk_0ccd885db9a4b5cfb29917c7800602571b1ba75ecc3c30e9")

# Convert text to speech (returns a generator)
audio_chunks = client.text_to_speech.convert(
    voice_id="EXAVITQu4vr4xnSDxMaL",
    model_id="eleven_monolingual_v1",
    text="Hello, this is a test of ElevenLabs Realtime API with chunked audio."
)

# Write chunks to a file
with open("test_output.mp3", "wb") as f:
    for chunk in audio_chunks:
        f.write(chunk)

print("✅ Audio generated successfully.")
