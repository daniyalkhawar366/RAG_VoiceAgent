import os
import sys
import requests
import json
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
from config import GOOGLE_STT_API_URL

load_dotenv()

class SpeechTranscriber:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.key = os.getenv("GOOGLE_STT_KEY")

        if not self.key:
            print(
                "\n[WARNING] GOOGLE_STT_KEY not found in environment variables or .env file.\n"
                "          Speech-to-text will not work. Please add it to your .env file.\n"
            )
            self.url = None
        else:
            self.url = (
                f"{GOOGLE_STT_API_URL}"
                f"?client=chromium&output=json&key={self.key}&lang=en-US"
            )
            print("[STT] Google Web Speech API client initialised.")

    def transcribe(self, wav_filepath: str) -> str | None:
        """
        Transcribes a WAV file by extracting the raw PCM bytes and sending
        them to the Google Web Speech API.

        Note: This uses the unofficial chromium/v2 endpoint which requires
        no billing setup. For production use, replace with Whisper API or
        Deepgram for custom-vocabulary support and reliability guarantees.
        """
        if not self.url:
            print("[Transcriber] Cannot transcribe: GOOGLE_STT_KEY is not set.")
            return None

        if not os.path.exists(wav_filepath):
            print(f"[Transcriber] Error: WAV file '{wav_filepath}' not found.")
            return None

        try:
            with open(wav_filepath, "rb") as f:
                audio_data = f.read()

            # Skip the 44-byte WAV header to get raw signed 16-bit PCM
            # (mono, little-endian), which is what the API expects.
            raw_pcm = audio_data[44:]

            headers = {"Content-Type": f"audio/l16; rate={self.sample_rate}"}
            response = requests.post(self.url, headers=headers, data=raw_pcm, timeout=10)

            if response.status_code != 200:
                print(f"[Transcriber] API Error: HTTP {response.status_code}")
                return None

            # Google returns multiple JSON objects separated by newlines.
            for line in response.text.split("\n"):
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "result" in data and data["result"]:
                            return data["result"][0]["alternative"][0]["transcript"]
                    except Exception:
                        pass
            return None
        except Exception as e:
            print(f"[Transcriber] Exception during transcription: {e}")
            return None
