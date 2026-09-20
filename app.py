import io
import os
import json
import base64
import tempfile
import subprocess
import boto3
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
from faster_whisper import WhisperModel
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Get bundled ffmpeg binary path (no system install needed)
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[INFO] Using bundled ffmpeg: {FFMPEG_EXE}")
except Exception as _e:
    FFMPEG_EXE = "ffmpeg"  # fallback to system ffmpeg
    print(f"[WARN] imageio-ffmpeg not found, using system ffmpeg: {_e}")

app = FastAPI(title="Real-Time AI Language Tutor")

# Ensure static folder exists & mount it
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
polly_client = boto3.client("polly", region_name=REGION)

# Initialize faster-whisper at server startup
# CUDA float16 on GPU (fastest), fallback to CPU int8 if CUDA is unavailable
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"[INFO] Initializing faster-whisper ('small') on {device.upper()} with {compute_type}...")
whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
print("[✓] Speech model loaded successfully!")

# ISO-639-1 Language mapping for faster-whisper
LANG_CODE_MAP = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "marathi": "mr",
    "bengali": "bn",
}


def get_lang_code(mode: str, target_lang: str, mother_tongue: str) -> str:
    """
    Return the Whisper ISO-639-1 language code.
    - 'practice' mode: user speaks target language
    - 'bridge' mode: user speaks their mother tongue
    """
    if mode == "bridge":
        key = mother_tongue.strip().lower()
    else:
        key = target_lang.strip().lower()
    return LANG_CODE_MAP.get(key, "en")


def convert_audio_to_wav(audio_bytes: bytes, filename: str) -> io.BytesIO:
    """
    Use ffmpeg subprocess directly (bypassing pydub) to convert any browser
    audio format (webm, ogg, mp4, wav …) to 16 kHz mono WAV.
    """
    suffix = os.path.splitext(filename)[1] or ".webm"

    # Write raw bytes to a temp input file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path + ".wav"

    try:
        result = subprocess.run(
            [
                FFMPEG_EXE,
                "-y",                   # overwrite output without asking
                "-i", input_path,       # input file
                "-ar", "16000",         # sample rate 16 kHz
                "-ac", "1",             # mono channel
                "-acodec", "pcm_s16le", # 16-bit PCM (standard WAV)
                "-f", "wav",            # force WAV container
                output_path,
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg conversion failed:\n{err}")

        with open(output_path, "rb") as f:
            wav_bytes = f.read()

        return io.BytesIO(wav_bytes)

    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    target_lang: str = Form("English"),
    mother_tongue: str = Form("Hindi"),
    mode: str = Form("practice")
):
    try:
        audio_bytes = await file.read()
        filename = file.filename or "voice.webm"

        print(f"[INFO] Received audio: {filename}, size={len(audio_bytes)} bytes, lang={target_lang}, mode={mode}")

        # Convert incoming audio to 16 kHz mono WAV
        try:
            wav_io = convert_audio_to_wav(audio_bytes, filename)
        except Exception as conv_err:
            print(f"[AUDIO CONVERT ERROR]: {conv_err}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Could not decode audio: {conv_err}"}
            )

        # Determine target ISO language code
        stt_lang = get_lang_code(mode, target_lang, mother_tongue)
        print(f"[INFO] Whisper target language code: '{stt_lang}'")

        # Transcribe with faster-whisper
        segments, info = whisper_model.transcribe(
            wav_io,
            language=stt_lang,
            beam_size=5,
            vad_filter=True, # Filters out background hum & repetitive stuttering
            vad_parameters=dict(min_silence_duration_ms=400)
        )

        user_transcript = " ".join([segment.text for segment in segments]).strip()

        # Fallback if no speech was detected in the specific target language
        if not user_transcript:
            wav_io.seek(0)
            segments_fb, _ = whisper_model.transcribe(wav_io, beam_size=3, vad_filter=True)
            user_transcript = " ".join([s.text for s in segments_fb]).strip()

        if not user_transcript:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not detect clear speech. Please hold the mic, speak clearly, and release."}
            )

        print(f"[INFO] Transcript: {user_transcript}")

        # Structured Assessment Prompt for Bedrock
        system_prompt = f"""You are 'Tom', an encouraging, expert real-time AI language tutor.
Target Language to Learn: {target_lang}
Learner's Mother Tongue: {mother_tongue}
Current Learning Mode: {mode} (either 'practice' where user speaks in target language, or 'bridge' where user speaks their mother tongue to learn how to say it in the target language).

Analyze the learner's spoken input and return ONLY a valid JSON object matching this schema exactly with no surrounding Markdown backticks:
{{
  "score": <integer from 40 to 100 representing accuracy/natural flow>,
  "detected_lang": "<language detected>",
  "original_text": "<verbatim transcript>",
  "corrected_text": "<ideal sentence in target language>",
  "explanation": "<1-2 short friendly sentences pointing out corrections, meaning, or pronunciation tips>",
  "phonetic_guide": "<Romanized pronunciation guide if target is non-English, or syllable guide if English>",
  "spoken_reply": "<what Tom will say out loud: 1 brief sentence praise/correction + 1 short engaging follow-up question>"
}}
"""

        bedrock_response = bedrock_client.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": f"User speech: '{user_transcript}'"}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"maxTokens": 300, "temperature": 0.4}
        )

        raw_output = bedrock_response['output']['message']['content'][0]['text'].strip()

        # Clean any accidental Markdown fence tags
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:]
        if raw_output.startswith("```"):
            raw_output = raw_output[3:]
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3]

        assessment_data = json.loads(raw_output.strip())

        # Synthesize Tom's voice via AWS Polly
        speech_text = assessment_data.get("spoken_reply", "Good effort! Let's keep practicing.")
        voice_id = "Kajal" if target_lang.lower() == "hindi" else "Joanna"

        polly_response = polly_client.synthesize_speech(
            Text=speech_text,
            OutputFormat="mp3",
            VoiceId=voice_id,
            Engine="neural"
        )
        audio_stream = polly_response["AudioStream"].read()
        audio_base64 = base64.b64encode(audio_stream).decode("utf-8")

        return {
            "success": True,
            "data": assessment_data,
            "audio_base64": audio_base64
        }

    except Exception as e:
        print(f"[ERROR]: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})