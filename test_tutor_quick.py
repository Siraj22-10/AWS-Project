import io
import boto3
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr

REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
SAMPLE_RATE = 16000
DURATION = 5  # Speaks for 5 seconds

print("==================================================")
print("   REAL-TIME AI LANGUAGE TUTOR - LIVE MIC TEST    ")
print("==================================================")

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
polly_client = boto3.client("polly", region_name=REGION)

# 1. Capture live audio from your microphone using sounddevice
print(f"\n[MIC] Listening... Speak a sentence into your microphone now ({DURATION} seconds)!")
audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
sd.wait()
print("[MIC] Audio captured. Converting speech to text...")

# 2. Convert recorded audio buffer into WAV for recognition
wav_io = io.BytesIO()
sf.write(wav_io, audio_data, SAMPLE_RATE, format='WAV', subtype='PCM_16')
wav_io.seek(0)

# 3. Speech Recognition
r = sr.Recognizer()
with sr.AudioFile(wav_io) as source:
    recorded_audio = r.record(source)

try:
    user_text = r.recognize_google(recorded_audio)
    print(f"\n>>> YOU SAID: \"{user_text}\"")
except Exception as e:
    print(f"\n[RECOGNITION ERROR]: {e}")
    user_text = "I am practicing speaking English today."
    print(f">>> FALLBACK TEXT: \"{user_text}\"")

# 4. AWS Bedrock: Language Tutor Feedback
print(f"\n[BEDROCK] Analyzing speech with {MODEL_ID}...")
response = bedrock_client.converse(
    modelId=MODEL_ID,
    messages=[{"role": "user", "content": [{"text": user_text}]}],
    system=[{"text": "You are a friendly, encouraging real-time English tutor. Politely point out any grammatical or phrasing errors in 1-2 short sentences, then ask an engaging follow-up question."}],
    inferenceConfig={"maxTokens": 120, "temperature": 0.5}
)
tutor_reply = response['output']['message']['content'][0]['text']
print(f"\n>>> TUTOR REPLY: \"{tutor_reply}\"")

# 5. AWS Polly: Neural voice speech
print("\n[POLLY] Synthesizing tutor voice...")
polly_resp = polly_client.synthesize_speech(
    Text=tutor_reply,
    OutputFormat="ogg_vorbis",
    VoiceId="Joanna",
    Engine="neural"
)

audio_stream = polly_resp["AudioStream"].read()
data, fs = sf.read(io.BytesIO(audio_stream))
print("\n[AUDIO] Playing tutor response through speakers...")
sd.play(data, fs)
sd.wait()

print("\n[SUCCESS] Full live microphone test complete!")