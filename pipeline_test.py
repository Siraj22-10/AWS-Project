import asyncio
import io
import boto3
import sounddevice as sd
import soundfile as sf
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

# ==========================================================
# CONFIGURATION
# ==========================================================
REGION = "us-east-1"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5   # Record for 5 seconds

# Using Amazon Nova Micro: Instant access, ultra-low latency, lowest cost
MODEL_ID = "amazon.nova-micro-v1:0"

# Initialize AWS SDK Clients
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
polly_client = boto3.client("polly", region_name=REGION)


# ----------------------------------------------------------
# 1. RECORD AUDIO FROM YOUR MICROPHONE
# ----------------------------------------------------------
def record_user_audio(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print(f"\n[MIC] Listening... Speak a sentence into your mic ({duration} seconds)!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("[MIC] Audio recording finished.")
    return audio_data.tobytes()


# ----------------------------------------------------------
# 2. AMAZON TRANSCRIBE (Speech-to-Text Streaming)
# ----------------------------------------------------------
class TutorTranscriptHandler(TranscriptResultStreamHandler):
    def __init__(self, output_stream):
        super().__init__(output_stream)
        self.full_transcript = []

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if not result.is_partial:
                for alt in result.alternatives:
                    self.full_transcript.append(alt.transcript)


async def transcribe_audio_stream(raw_audio_bytes):
    print("[TRANSCRIBE] Converting speech to text via AWS...")
    client = TranscribeStreamingClient(region=REGION)
    stream = await client.start_stream_transcription(
        language_code="en-US",
        media_sample_rate_hz=SAMPLE_RATE,
        media_encoding="pcm"
    )

    handler = TutorTranscriptHandler(stream.output_stream)

    async def send_audio():
        chunk_size = 1024
        for i in range(0, len(raw_audio_bytes), chunk_size):
            chunk = raw_audio_bytes[i:i + chunk_size]
            await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await asyncio.sleep(0.01)
        await stream.input_stream.end_stream()

    await asyncio.gather(send_audio(), handler.handle_events())
    final_text = " ".join(handler.full_transcript).strip()
    print(f"[YOU SAID]: \"{final_text}\"")
    return final_text


# ----------------------------------------------------------
# 3. AMAZON BEDROCK (AI Language Tutor Response)
# ----------------------------------------------------------
def get_tutor_feedback(user_text):
    if not user_text:
        return "I couldn't hear any speech. Could you please try speaking again?"

    print(f"[BEDROCK] Generating tutor response using {MODEL_ID}...")
    system_prompt = (
        "You are an encouraging, expert real-time English language tutor. "
        "The user is practicing their spoken English. "
        "Analyze the user's sentence for grammar mistakes. "
        "Keep your response concise (maximum 2 sentences). "
        "Briefly provide a friendly correction if needed, then ask an engaging follow-up question."
    )

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": user_text}]
            }
        ],
        system=[{"text": system_prompt}],
        inferenceConfig={
            "maxTokens": 120,
            "temperature": 0.5
        }
    )

    tutor_reply = response['output']['message']['content'][0]['text']
    print(f"[TUTOR REPLY]: \"{tutor_reply}\"")
    return tutor_reply


# ----------------------------------------------------------
# 4. AMAZON POLLY (Neural Text-to-Speech Playback)
# ----------------------------------------------------------
def speak_tutor_response(reply_text):
    print("[POLLY] Synthesizing Neural voice response...")
    response = polly_client.synthesize_speech(
        Text=reply_text,
        OutputFormat="ogg_vorbis",
        VoiceId="Joanna",
        Engine="neural"
    )

    if "AudioStream" in response:
        audio_stream = response["AudioStream"].read()
        data, fs = sf.read(io.BytesIO(audio_stream))
        print("[AUDIO] Speaking feedback through your speakers...")
        sd.play(data, fs)
        sd.wait()
        print("[AUDIO] Playback complete.")


# ----------------------------------------------------------
# MAIN ORCHESTRATOR
# ----------------------------------------------------------
async def main():
    print("==================================================")
    print("   REAL-TIME AI LANGUAGE TUTOR - TEST PIPELINE    ")
    print("==================================================")

    # 1. Record your voice
    audio_bytes = record_user_audio(duration=5)

    # 2. STT via Amazon Transcribe
    transcript = await transcribe_audio_stream(audio_bytes)

    # 3. AI Tutor reasoning via Bedrock
    tutor_reply = get_tutor_feedback(transcript)

    # 4. Voice synthesis via Amazon Polly
    speak_tutor_response(tutor_reply)

    print("\n[SUCCESS] Entire loop completed without errors!")

if __name__ == "__main__":
    asyncio.run(main())