# Real-Time AI Language Tutor

## 📖 Project Overview
The **Real-Time AI Language Tutor** (featuring "Tom", the virtual teacher) is an interactive, web-based educational application designed to help users learn and practice various languages. The application provides a virtual classroom environment where users can speak into their microphone, and the AI will analyze their speech, correct grammar, provide phonetic guides, and respond with synthesized voice feedback.

The application supports multiple Indian and global languages, including English, Hindi, Telugu, Tamil, Kannada, Marathi, and Bengali. It features two primary learning modes:
1. **Practice Mode**: The user speaks directly in the target language they are trying to learn.
2. **Bridge Mode**: The user speaks in their mother tongue, and the AI helps them translate and learn how to say it in the target language.

## 🛠️ Technology Stack

### Backend
*   **Python (FastAPI)**: The core backend framework, chosen for its high performance and async capabilities. It handles routing, API endpoints, and static file serving.
*   **faster-whisper**: An optimized implementation of OpenAI's Whisper model used for **local Speech-to-Text (STT)** transcription. It processes the user's audio input. It leverages CUDA for GPU acceleration if available, falling back to CPU (int8) otherwise.
*   **FFmpeg (via imageio-ffmpeg)**: Used to convert various browser audio formats (like WebM or OGG) into standard 16 kHz mono WAV files required by the Whisper model.
*   **Boto3**: The official AWS SDK for Python, used to interact seamlessly with Amazon Bedrock and Amazon Polly.
*   **Jinja2**: Templating engine used by FastAPI to serve the frontend HTML.

### Frontend
*   **HTML5 / CSS3 / Vanilla JavaScript**: A custom-built, lightweight frontend without heavy frameworks.
*   **MediaRecorder API**: Native browser API used to capture the user's voice through their microphone.
*   **UI/UX Design**: Features a highly stylized, aesthetic "virtual classroom" design with animations (chalkboard writing, avatar breathing/blinking, dynamic scoreboards) using CSS variables and keyframes.
*   **Fonts & Icons**: Utilizes Google Fonts (Nunito, Caveat, DM Sans) to simulate handwriting on the chalkboard, and FontAwesome for UI icons.

## ☁️ AWS Services Integration
The project relies heavily on Amazon Web Services (AWS) for its core AI reasoning and voice generation capabilities. Here is a pin-to-pin breakdown of the AWS services utilized:

### 1. Amazon Bedrock
*   **Role**: The "Brain" of the AI Tutor.
*   **Model Used**: `amazon.nova-micro-v1:0` (Amazon Nova Micro).
*   **Implementation Details**:
    *   Once `faster-whisper` transcribes the user's audio into text, the backend constructs a detailed prompt.
    *   The prompt instructs the Bedrock model to act as "Tom", the language tutor. It provides the target language, mother tongue, learning mode, and the transcribed text.
    *   Bedrock is strictly instructed to return a **JSON object** (bypassing markdown formatting).
    *   **Data Analyzed & Returned**: 
        *   `score`: An integer rating (40-100) on accuracy and flow.
        *   `corrected_text`: The grammatically ideal sentence.
        *   `explanation`: A brief, friendly tip or correction.
        *   `phonetic_guide`: A Romanized pronunciation guide.
        *   `spoken_reply`: The exact text the AI tutor will say back to the user.

### 2. Amazon Polly
*   **Role**: The "Voice" of the AI Tutor (Text-to-Speech).
*   **Implementation Details**:
    *   After Bedrock generates the `spoken_reply`, the text is sent to Amazon Polly.
    *   **Engine**: Configured to use the **Neural** engine for the most natural, human-like voice synthesis.
    *   **Dynamic Voice Selection**: The backend dynamically selects the Voice ID based on the target language. For example, if the target language is Hindi, it uses the Indian voice **"Kajal"**. For other languages, it defaults to **"Joanna"**.
    *   **Audio Delivery**: Polly returns an MP3 audio stream. The Python backend reads this stream, converts it to a Base64-encoded string, and sends it back to the frontend API response. The frontend then decodes and automatically plays the audio using the HTML5 `Audio` object.

## 🚀 How It Works (The Pipeline)
1. **Capture**: The user holds the microphone button on the web interface and speaks.
2. **Upload**: The browser sends the recorded audio (usually `.webm`) to the `/api/process-audio` endpoint.
3. **Pre-process**: The backend uses `ffmpeg` to convert the audio to a 16kHz WAV format.
4. **Transcribe**: `faster-whisper` transcribes the audio into text.
5. **Analyze (AWS)**: The text is sent to **Amazon Bedrock**, which analyzes the grammar, scores it, and generates a response in JSON format.
6. **Synthesize (AWS)**: The `spoken_reply` from Bedrock is sent to **Amazon Polly**, which generates a Neural TTS MP3 audio file.
7. **Display & Play**: The frontend receives the JSON data and Base64 audio, updates the chalkboard UI with the corrections, and plays the AI tutor's voice.
