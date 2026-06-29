"""
config.py — Central configuration for the Genesis CPO RAG Voice Agent.

All tunable parameters are defined here so that no magic numbers are
scattered across the codebase. Change a value here and it propagates
everywhere automatically.
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GOOGLE_STT_API_URL = "https://www.google.com/speech-api/v2/recognize"

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
LLM_PRIMARY_MODEL   = "llama-3.3-70b-versatile"
LLM_FALLBACK_MODEL  = "llama-3.1-8b-instant"
LLM_TEMPERATURE     = 0.0          # Set to 0.0 for strict factual adherence (prevents price hallucinations)
LLM_INTENT_TEMP     = 0.0          # Zero temperature for deterministic intent classification
LLM_MAX_TOKENS      = 350          # Max tokens per streamed response (350 avoids mid-sentence cutoff)
LLM_INTENT_TOKENS   = 15           # Max tokens for intent classification (single-word output with headroom)
LLM_FILTER_TOKENS   = 120          # Max tokens for filter/rewrite extraction (avoids JSON truncation)
LLM_RESPONSE_TIMEOUT = 8           # Seconds before Groq API call times out
LLM_FAST_TIMEOUT     = 5           # Seconds for lightweight calls (intent, rewrite, filter)

# ---------------------------------------------------------------------------
# RAG / Retrieval
# ---------------------------------------------------------------------------
RAG_TOP_K                = 5       # Number of results returned per query
RAG_MODEL_FILTER_LIMIT   = 25      # Wider pool fetched when a model-family filter is active
RAG_PRICE_RANGE_MARGIN   = 0.15    # ± margin for "around X" price queries (15%)
RAG_SIMILARITY_THRESHOLD = 1.40    # Maximum L2 distance kept for PURE semantic searches.
                                   # This gate is intentionally skipped when a metadata
                                   # where_clause (price range, body type) is active —
                                   # a rewritten price query like "price between 200000 and 210000"
                                   # is semantically distant from car descriptions even for
                                   # perfectly correct matches, so the filter itself is the
                                   # correctness mechanism in that path.
                                   # 1.40 allows through generic intent queries ("family car",
                                   # "sporty", "electric") while still blocking truly off-brand
                                   # queries (Ferrari, Toyota) that embed very far from Genesis docs.
RAG_FUZZY_MATCH_SCORE    = 70      # Minimum rapidfuzz partial_ratio score to accept an active-car match
RAG_HISTORY_WINDOW       = 20      # Max conversation turns passed to LLM

# ---------------------------------------------------------------------------
# Audio Recording (VAD)
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE       = 16000    # Hz — matches Google STT expectation
AUDIO_CHUNK_SIZE        = 1024     # Frames per read chunk
AUDIO_THRESHOLD         = 0.010    # Default RMS energy floor (overridden by dynamic calibration)
AUDIO_SILENCE_DURATION  = 1.4      # Seconds of silence before recording stops
AUDIO_NO_SPEECH_TIMEOUT = 5.0      # Seconds without any speech before giving up
AUDIO_NOISE_MARGIN      = 1.8      # Multiplier applied to ambient noise floor
AUDIO_NOISE_CAP         = 0.08     # Hard upper cap on the dynamic threshold

# ---------------------------------------------------------------------------
# TTS (edge-tts)
# ---------------------------------------------------------------------------
TTS_VOICE = "en-US-AndrewNeural"              # Warm, confident, premium male voice
TTS_RATE  = "-2%"                           # Natural tempo, clear and professional

# ---------------------------------------------------------------------------
# Dynamic Initialization
# ---------------------------------------------------------------------------
# Extract available model families from inventory on startup
VALID_MODELS = set()
try:
    with open(os.path.join(BASE_DIR, "data", "cars.json"), "r", encoding="utf-8") as f:
        cars = json.load(f)
        for car in cars:
            # e.g., "GV80 3.5 Royal" -> "GV80"
            name_parts = car.get("name", "").split()
            if name_parts:
                VALID_MODELS.add(name_parts[0].upper())
except Exception as e:
    print(f"[Config Warning] Could not load cars.json to extract model families: {e}")
    VALID_MODELS = {"GV80", "G80", "G90", "GV70", "G70", "GV60"}

AVAILABLE_MODELS_LIST = list(VALID_MODELS)
