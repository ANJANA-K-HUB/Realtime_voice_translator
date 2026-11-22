
# 🎙️ Real-Time AI Voice Translator

A browser-based **Real-Time AI Voice Translator** that converts spoken language into another language using Google's **Gemini Live API**. Built with Streamlit, WebRTC, PyAV, NumPy, and Python, the application captures microphone audio, streams it to Gemini for real-time translation, displays live transcripts, and allows users to play back the translated audio.

## 🚀 Key Features

- **Real-Time Speech Translation:** Captures microphone audio and streams it to the Gemini Live API for real-time speech translation.
- **Live Transcript Streaming:** Displays both the original speech transcript and translated text while the user is speaking.
- **Audio Processing:** Converts browser microphone audio from 48 kHz to 16 kHz PCM16 audio chunks (10 ms) before sending it to Gemini.
- **Translated Audio Playback:** Keeps audio output silent during recording and allows users to play captured translated audio after clicking STOP.
- **72 Target Languages:** Supports all 22 official languages listed in India's Eighth Schedule, along with 50 additional languages.
- **Round-Based Recording:** Supports multiple recording sessions. Starting a new round clears the previous round's captured audio.
- **Session Resilience:** Handles Gemini session rotations (`go_away`), interruptions, and quota-related backoff mechanisms.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| streamlit-webrtc | Browser-based microphone audio streaming |
| PyAV | Audio frame processing |
| NumPy | Audio data manipulation and processing |
| Google Gemini Live API | Real-time speech translation |
| Asyncio | Asynchronous background processing |
| python-dotenv | Environment variable and API key management |

## 📁 Project Structure

```text
Real-Time-AI-Voice-Translator/
│
├── app.py                     # Streamlit UI, WebRTC audio processor,
│                              # round playback and .env loader
│
├── live_translator_worker.py  # Background asyncio worker for Gemini
│                              # Live API streaming and transcripts
│
├── audio_utils.py             # Audio resampling and chunking helpers
│
├── .env.example               # Environment variable template
│
├── .gitignore                 # Excludes sensitive files and artifacts
│
└── README.md                  # Project documentation
```

## ⚙️ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone "https://github.com/ANJANA-K-HUB/Realtime_voice_translator"
```

Navigate to the project directory:

```bash
cd Real-Time-AI-Voice-Translator
```

### Step 2: Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install streamlit streamlit-webrtc av numpy google-genai python-dotenv
```

### Step 4: Configure the Gemini API Key

The application requires a Gemini API key to access the Gemini Live API.

1. Visit [Google AI Studio](https://aistudio.google.com/api-keys).
2. Generate your own Gemini API key.
3. Locate the `.env.example` file in the project directory.
4. Create a copy named `.env`.

For Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

5. Open the `.env` file and add your API key:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

**Important:**
- Replace the placeholder with your actual Gemini API key.
- Do not add spaces around the `=` symbol.
- Never upload your `.env` file to GitHub.
- Each user should configure their own API key.

### Step 5: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

## 🎯 How to Use the Application

1. Open the application in your browser.
2. Select your preferred **Target Language** from the sidebar.
3. Click the **START** button.
4. Allow microphone access when prompted by the browser.
5. Speak clearly into your microphone.
6. Observe the original and translated transcripts in the Live Transcript section.
7. Click **STOP** when you finish speaking.
8. Click **Play Captured Gemini Audio** to listen to the translated speech from that recording session.
9. Click **START** again to begin a new recording round.

**Note:** The application keeps the translated speaker output silent during recording. Translated audio is played only after the recording session is stopped and the playback button is clicked.

## 🔄 Application Workflow

The application follows this audio-processing and translation pipeline:

```text
        User Speaks
             |
             v
     Browser Microphone
             |
             v
       WebRTC Streaming
             |
             v
      Audio Frame Capture
             |
             v
      PyAV + NumPy Processing
             |
             v
    48 kHz to 16 kHz Conversion
             |
             v
       PCM16 Audio Chunks
             |
             v
       Gemini Live API
             |
             v
     Speech Recognition
             |
             v
      Language Translation
             |
       +-----+------+
       |            |
       v            v
 Original Text   Translated Text
       |            |
       +-----+------+
             |
             v
       Streamlit UI
             |
             v
        STOP Recording
             |
             v
    Play Captured Gemini Audio
```

## 🔐 Security Best Practices

- Keep API keys and other sensitive credentials in the `.env` file.
- Add `.env` to `.gitignore`.
- Commit `.env.example` instead of `.env`.
- Never expose your Gemini API key in source code or public repositories.
- Each developer or student should generate and configure their own API key.

Example `.gitignore`:

```gitignore
# Environment variables
.env

# Virtual environment
venv/
.venv/

# Python cache
__pycache__/
*.pyc

# IDE files
.vscode/
.idea/

# Operating system files
.DS_Store
Thumbs.db
```

## 🌐 Supported Languages

The application supports **72 target languages**:

- 22 official languages included in the Eighth Schedule of the Indian Constitution.
- 50 additional languages from different parts of the world.

Users can select their desired target language through the application sidebar.

## 📌 Future Enhancements

Potential improvements for future development include:

- Support for additional language pairs.
- Improved translation accuracy and latency.
- Downloadable translated audio recordings.
- Translation history and session management.
- User authentication and personalized settings.
- Cloud deployment for public access.
- Enhanced error handling and connection monitoring.

## 👨‍💻 Project Information

| Field | Details |
|---|---|
| Project Name | Real-Time AI Voice Translator |
| Category | Artificial Intelligence, Speech Processing, Real-Time Translation |
| Core API | Google Gemini Live API |
| Frontend | Streamlit |
| Programming Language | Python |

---

⭐ If you find this project useful, consider giving the repository a star!
