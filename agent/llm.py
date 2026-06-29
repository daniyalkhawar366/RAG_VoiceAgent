import os
import re
import sys
import requests
import json
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()

# Import central config — fall back to safe defaults if run standalone
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
try:
    from config import (
        LLM_PRIMARY_MODEL, LLM_FALLBACK_MODEL,
        LLM_TEMPERATURE, LLM_INTENT_TEMP,
        LLM_MAX_TOKENS, LLM_INTENT_TOKENS, LLM_FILTER_TOKENS,
        LLM_RESPONSE_TIMEOUT, LLM_FAST_TIMEOUT,
        GROQ_API_URL, PROMPTS_DIR, AVAILABLE_MODELS_LIST
    )
except ImportError:
    LLM_PRIMARY_MODEL    = "llama-3.3-70b-versatile"
    LLM_FALLBACK_MODEL   = "llama-3.1-8b-instant"
    LLM_TEMPERATURE      = 0.0
    LLM_INTENT_TEMP      = 0.0
    LLM_MAX_TOKENS       = 250
    LLM_INTENT_TOKENS    = 10
    LLM_FILTER_TOKENS    = 60
    LLM_RESPONSE_TIMEOUT = 8
    LLM_FAST_TIMEOUT     = 5


def load_prompt(filename):
    try:
        with open(os.path.join(PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[LLM Warning] Could not load prompt {filename}: {e}")
        return ""

class LLMAgent:
    def __init__(self):
        # Load all available API keys for round-robin rotation
        all_keys = [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY2"),
            os.getenv("GROQ_API_KEY3"),
        ]
        self._api_keys = [k for k in all_keys if k]  # filter out missing keys
        self._key_index = 0                           # pointer into the key list
        self.api_key    = self._api_keys[0] if self._api_keys else None
        self.model_name = LLM_PRIMARY_MODEL
        self.history    = []

        if not self.api_key:
            print("\n[WARNING] No GROQ_API_KEY found in environment variables or .env file.")
            print("The agent will fall back to a mock mode. Please verify your .env file.\n")
        else:
            print(f"[LLM] Groq API initialized with {len(self._api_keys)} key(s), model '{self.model_name}'.")

    def _next_key(self) -> str:
        """Rotate to the next available API key (round-robin)."""
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        new_key = self._api_keys[self._key_index]
        print(f"[LLM] Rotating to API key index {self._key_index}.")
        return new_key

    def _post_with_retry(self, url: str, headers: dict, payload: dict, timeout: int, stream: bool = False):
        """
        POST to Groq with automatic key rotation and model fallback.
        Tries each key once before falling back to the smaller model.
        Returns (response, final_model_used).
        """
        tried_keys = set()
        current_key = self.api_key

        while True:
            headers["Authorization"] = f"Bearer {current_key}"
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout, stream=stream)
            except Exception as e:
                raise e

            if resp.status_code == 200:
                return resp, payload["model"]

            if resp.status_code == 429:
                tried_keys.add(current_key)
                # Try next key first
                remaining_keys = [k for k in self._api_keys if k not in tried_keys]
                if remaining_keys:
                    current_key = remaining_keys[0]
                    # Advance internal pointer to match
                    self._key_index = self._api_keys.index(current_key)
                    print(f"[LLM] Rate limited. Rotating to key index {self._key_index}.")
                    continue
                # All keys exhausted for this model — fallback to smaller model
                if payload["model"] == LLM_PRIMARY_MODEL:
                    payload["model"] = LLM_FALLBACK_MODEL
                    tried_keys.clear()  # reset key rotation for the fallback model
                    current_key = self._api_keys[0]
                    print(f"[LLM] All keys rate-limited on primary model. Falling back to {LLM_FALLBACK_MODEL}.")
                    continue
            # Non-429 error — return as-is so caller can handle it
            return resp, payload["model"]
            
    def classify_intent(self, query: str, active_car_name: str) -> str:
        """
        Classifies the customer's intent into one of these categories:
        INVENTORY_SEARCH, FEATURE_QUESTION, PRICE_QUESTION, COMPARISON, AFFIRMATION, SMALLTALK, EXIT
        """
        if not self.api_key:
            # Fallback mock mode
            query_lower = query.lower()
            if any(cmd in query_lower for cmd in ["goodbye", "bye-bye", "bye", "exit", "quit", "stop", "hang up", "done"]):
                return "EXIT"
            if any(w in query_lower for w in ["yes", "sure", "ok", "okay", "go ahead"]):
                return "AFFIRMATION"
            if any(w in query_lower for w in ["compare", "difference", "vs"]):
                return "COMPARISON"
            if any(w in query_lower for w in ["hello", "hi", "how are you", "who are you"]):
                return "SMALLTALK"
            if any(w in query_lower for w in ["don't understand", "confused", "explain", "what do you mean"]):
                return "CONFUSION"
            if any(w in query_lower for w in ["summarize", "which one did", "what did i like", "summary", "recap"]):
                return "SUMMARY"
            
            # OUT_OF_SCOPE checks (including adversarial attacks, financing/leasing)
            out_of_scope_keywords = [
                "system prompt", "chatgpt", "vector database", "api key", "joke", "weather", 
                "sports", "bitcoin", "programming", "code", "world cup", "math", "calculator",
                "reveal your", "act as", "bypass instructions", "finance", "financing", "payment", 
                "lease", "apr", "interest rate"
            ]
            if any(w in query_lower for w in out_of_scope_keywords):
                return "OUT_OF_SCOPE"
            
            # NEEDS_CONTEXT checks: contains pronouns referencing an unnamed vehicle with no active car
            has_pronoun = any(w in query_lower.split() for w in ["it", "this", "that", "it's", "its"])
            is_property_query = any(w in query_lower for w in ["feature", "spec", "color", "sunroof", "leather", "carplay", "android", "price", "cost", "how much", "massage"])
            
            # If the query is rich descriptive search criteria, it's INVENTORY_SEARCH
            has_search_criteria = any(w in query_lower for w in ["under", "budget", "recommend", "looking for", "family car", "show me", "i want"])
            
            if active_car_name.lower() in ["none", "unknown", ""] and is_property_query and has_pronoun and not has_search_criteria:
                return "NEEDS_CONTEXT"
                
            if any(w in query_lower for w in ["price", "cost", "how much", "budget", "sar"]):
                return "PRICE_QUESTION"
            if any(w in query_lower for w in ["feature", "spec", "color", "sunroof", "leather", "carplay", "android"]):
                return "FEATURE_QUESTION"
            return "INVENTORY_SEARCH"

        template = load_prompt("classify_intent.txt")
        models_str = ", ".join(AVAILABLE_MODELS_LIST) if AVAILABLE_MODELS_LIST else "GV80, G80, G90, GV70, G70, GV60"
        prompt = template.format(
            AVAILABLE_MODELS_LIST_STR=models_str,
            active_car_name=active_car_name,
            query=query
        )

        url = GROQ_API_URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_INTENT_TOKENS,
            "temperature": LLM_INTENT_TEMP
        }

        try:
            response, _ = self._post_with_retry(url, headers, payload, LLM_FAST_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip().upper()
            else:
                return "INVENTORY_SEARCH"
        except Exception as e:
            print(f"[LLM Exception] Failed to classify intent: {e}")
            return "INVENTORY_SEARCH"

    def _format_cars(self, retrieved_cars):
        parts = []
        for idx, res in enumerate(retrieved_cars):
            doc = res.get('document', '')
            url = res.get('metadata', {}).get('url', 'N/A')
            parts.append(f"[Match {idx+1}] {doc} (URL: {url})")
        return "\n".join(parts)

    def extract_filters(self, query: str) -> dict:
        """
        Extracts structured filters from a search query using the LLM.
        Returns a dict with optional keys: model_family, body_type, max_price, min_price, color, fuel_type
        """
        if not self.api_key:
            return {}

        template = load_prompt("extract_filters.txt")
        models_str = ", ".join([f'"{m}"' for m in AVAILABLE_MODELS_LIST]) if AVAILABLE_MODELS_LIST else '"GV80", "G80", "G90", "GV70", "G70", "GV60"'
        prompt = template.format(
            AVAILABLE_MODELS_LIST_STR=models_str,
            query=query
        )

        url = GROQ_API_URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": LLM_FILTER_TOKENS,
            "temperature": LLM_INTENT_TEMP
        }

        try:
            response, _ = self._post_with_retry(url, headers, payload, LLM_FAST_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.IGNORECASE)
                return json.loads(content)
            return {}
        except Exception as e:
            print(f"[LLM Error] Failed to extract filters: {e}")
            return {}

    def rewrite_query_if_needed(self, customer_query: str, session) -> str:
        """
        Rewrites the query on every turn to normalize STT errors and construct clean search queries.
        """
        if not self.api_key:
            return customer_query

        # Get active car summary
        active_car_str = "none"
        if session.active_car:
            m = session.active_car['metadata']
            features = ', '.join(m.get('features', []))
            active_car_str = f"{m['name']} | {m['price_sar']:,} SAR | Exterior: {m['exterior_color']} | Interior: {m['interior_color']} | Engine: {m['engine']} | Body: {m.get('body_type', 'unknown')} | Features: {features}"

        # Get last 4 turns of history
        history_snippet = []
        for m in session.history[-4:]:
            history_snippet.append(f"{m['role'].upper()}: {m['content']}")
        history_str = "\n".join(history_snippet) if history_snippet else "No history yet."

        rewrite_prompt = f"""You are a search query normalizer and rewriter for a Genesis car dealership inventory system.

Your job has two parts:
1. Fix any speech-to-text transcription errors, phonetic misspellings, or typos
2. Rewrite the query as a clear, standalone search string using conversation context

=== PHONETIC NORMALIZATION RULES ===
Always apply these corrections before rewriting:
- "Gee vee eighty" / "GV eighty" / "GV 80" / "GB 80" / "GD 80" → "GV80"
- "G eighty" / "G 80" / "GB80" / "GD80" → "G80"  
- "G ninety" / "G 90" → "G90"
- "G seventy" / "G 70" → "G70"
- "Gee vee seventy" / "GV seventy" → "GV70"
- "Gene sis" / "Jenesis" / "Genisis" → "Genesis"
- Any phonetic number spelling → numeral (e.g. "three point five" → "3.5")

=== PRICE NORMALIZATION RULES ===
Always restate price constraints in this standardized form:
- "under 200k" / "less than 200" / "below 200 thousand" → "price under 200000"
- "over 300k" / "more than 300" / "above 300 thousand" → "price over 300000"
- "between 200 and 300k" / "from 200 to 300" → "price between 200000 and 300000"
- "around 250k" / "about 250 thousand" → "price around 250000"
- "priced at most 200k" / "maximum 200" → "price under 200000"
- "entry level" / "cheapest" / "most affordable" / "least expensive" → "cheapest"
- "most expensive" / "top of the range" / "most premium" / "flagship" → "most expensive"

=== AMBIGUITY RULE ===
If the follow-up query is vague (e.g. "the blue one", "a Royal", "the electric one") 
and multiple options could match, write a BROAD query that retrieves all candidates.
Do NOT guess a specific model. Let the LLM see all options and ask a clarifying question.
Example: "the blue one" → "blue Genesis cars" (not "blue GV80")
Example: "the Royal" → "Genesis Royal trim" (not "GV80 Royal")
Example: "the electric one" → "Genesis electric EV" (not "G80 EV")

=== NEGATION RULE ===
If the customer EXPLICITLY REJECTS a body type, do NOT include it in the rewritten query.
Negation signals: "no SUVs", "not an SUV", "SUVs too expensive", "SUVs are too expensive",
"not a sedan", "sedans too expensive", "scratch the [type]", "forget [type]".
In these cases, write a BROAD query without any body type.
Example: "no SUVs are too expensive, suggest something in my budget" → "Genesis price under 120000"
Example: "not interested in sedans" → "Genesis cars"

=== CONTEXT ===
Currently discussed vehicle: {active_car_str}
Active filters: {session.active_filters}

Conversation history (last 4 turns):
{history_str}

Follow-up query to rewrite: {customer_query}

Output a SHORT search string of 3-10 words only. Apply all normalization rules above.
Return ONLY the rewritten query string, nothing else."""

        url = GROQ_API_URL
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": rewrite_prompt}],
            "max_tokens": LLM_FILTER_TOKENS,
            "temperature": LLM_INTENT_TEMP
        }

        try:
            response, _ = self._post_with_retry(url, headers, payload, LLM_FAST_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                rewritten = result["choices"][0]["message"]["content"].strip()
                rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
                print(f"[Memory] Query rewritten: '{customer_query}' → '{rewritten}'")
                return rewritten
            else:
                return customer_query
        except Exception as e:
            print(f"[LLM Exception] Failed to rewrite query: {e}")
            return customer_query

    def generate_response(self, customer_query, retrieved_contexts, history=None):
        """
        Generates a full sales response (non-streaming fallback).
        """
        if not self.api_key:
            return self._mock_response(customer_query, retrieved_contexts)
            
        context_str = self._build_context_str(retrieved_contexts)
        user_message = f"{context_str}\n\nCustomer Question: {customer_query}\n\nPlease respond to the customer query based on the live inventory context above."
        
        url = GROQ_API_URL
        headers = {"Content-Type": "application/json"}

        sys_prompt_template = load_prompt("agent_system_prompt.txt")
        sys_prompt = sys_prompt_template.format(session_context="No specific vehicle in focus yet.", inventory_context="No new inventory retrieved this turn.")
        messages = [{"role": "system", "content": sys_prompt}]

        chat_history = history if history is not None else self.history
        if chat_history:
            if chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == customer_query:
                history_to_append = chat_history[:-1]
            else:
                history_to_append = chat_history
            for hist_msg in history_to_append:
                messages.append(hist_msg)

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS
        }

        try:
            response, _ = self._post_with_retry(url, headers, payload, LLM_RESPONSE_TIMEOUT)
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"].strip()
                if history is None:
                    self.history.append({"role": "user", "content": customer_query})
                    self.history.append({"role": "assistant", "content": response_text})
                return response_text
            else:
                print(f"[LLM Error] Groq API returned status {response.status_code}: {response.text}")
                return "I apologize, but I am having trouble connecting to our database right now. Could you please repeat that?"
        except Exception as e:
            print(f"[LLM Exception] Failed to call Groq API: {e}")
            return "I apologize, but I am having trouble connecting to our database right now. Could you repeat that?"

    def generate_response_stream(self, customer_query: str, retrieved_cars: list, history: list = None, session_summary: str = "", active_car_name: str = "none", intent: str = ""):
        """
        Streams response tokens in real-time. Yields chunks of text as they arrive.
        """
        if not self.api_key:
            yield self._mock_response(customer_query, retrieved_cars)
            return

        if intent == "NEEDS_CONTEXT":
            system_prompt = """You are Alex, a Genesis Certified Pre-Owned sales consultant on a phone call.
The customer is asking about a specific vehicle feature but hasn't said which car yet.
Ask warmly: "I'd love to help with that! Which Genesis model did you have in mind?"
Nothing else.
"""
        else:
            template = load_prompt("agent_system_prompt.txt")
            session_ctx = session_summary if session_summary else "No specific vehicle in focus yet."
            inventory_ctx = self._format_cars(retrieved_cars) if retrieved_cars else "No new inventory retrieved this turn — use SESSION CONTEXT above."
            system_prompt = template.format(
                session_context=session_ctx,
                inventory_context=inventory_ctx
            )

        url = GROQ_API_URL
        headers = {"Content-Type": "application/json"}

        chat_history = history or []
        if chat_history and chat_history[-1]["role"] == "user" and chat_history[-1]["content"] == customer_query:
            history_to_append = chat_history[:-1]
        else:
            history_to_append = chat_history

        if retrieved_cars and intent != "NEEDS_CONTEXT":
            context_str = self._build_context_str(retrieved_cars)
            user_content = f"{context_str}\n\nCustomer Question: {customer_query}\n\nPlease respond based strictly on the live inventory context above. Never modify prices or invent features."
        else:
            user_content = customer_query

        messages = [
            {"role": "system", "content": system_prompt},
            *history_to_append,
            {"role": "user", "content": user_content}
        ]

        if intent == "NEEDS_CONTEXT":
            messages.append({
                "role": "system",
                "content": "IMPORTANT: The customer is asking a vehicle-specific question but no vehicle is currently in focus. You MUST reply with EXACTLY: \"I'd be happy to help with that. Which Genesis model were you asking about?\" Do not provide any other answer."
            })
        elif retrieved_cars:
            messages.append({
                "role": "system",
                "content": "CRITICAL DIRECTION: You are strictly forbidden from recommending, proposing, or mentioning any vehicle models, years, prices, colors, or specifications that are not explicitly listed in the RETRIEVED LIVE INVENTORY MATCHES context above. If no vehicle in the inventory meets their budget, recommend the closest match from the inventory list, state its real price, and EXPLICITLY state it exceeds the customer's budget. Do not invent any vehicles or prices. Stick strictly to the available inventory in context."
            })

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
            "stream": True
        }

        full_response = []
        try:
            response, _ = self._post_with_retry(url, headers, payload, LLM_RESPONSE_TIMEOUT, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(data_str)
                                delta = chunk_json["choices"][0]["delta"]
                                if "content" in delta:
                                    content = delta["content"]
                                    full_response.append(content)
                                    yield content
                            except Exception:
                                pass
                
                # Update internal history fallback
                if history is None:
                    response_text = "".join(full_response).strip()
                    self.history.append({"role": "user", "content": customer_query})
                    self.history.append({"role": "assistant", "content": response_text})
            else:
                print(f"[LLM Error] Groq API returned status {response.status_code}: {response.text}")
                yield "I apologize, but I am having trouble connecting to our database right now."
        except Exception as e:
            print(f"[LLM Exception] Failed to call Groq API: {e}")
            yield "I apologize, but I am having trouble connecting to our database right now."

    def _build_context_str(self, retrieved_contexts):
        context_str = "RETRIEVED LIVE INVENTORY MATCHES:\n"
        if retrieved_contexts:
            for idx, res in enumerate(retrieved_contexts):
                context_str += f"\n[Car Match {idx+1}]\n"
                context_str += f"Description: {res['document']}\n"
                context_str += f"Detail Link: {res['metadata'].get('url', 'N/A')}\n"
        else:
            context_str += "No live inventory matches found for this query.\n"
        return context_str

    def _mock_response(self, customer_query, retrieved_contexts):
        q = customer_query.lower()
        if not retrieved_contexts:
            return "We currently do not have that specific car in our live inventory. However, we have a range of premium Genesis models. What body style or features are you looking for?"
            
        car = retrieved_contexts[0]['metadata']
        name = car['name']
        price = car['price_sar']
        ext_color = car['exterior_color']
        
        if "price" in q or "cost" in q or "how much" in q:
            return f"The Genesis {name} in beautiful {ext_color} is available for {price:,} Saudi Riyals. It is a fantastic CPO vehicle. Would you like me to reserve it for you?"
        elif "color" in q or "look like" in q:
            return f"This {name} features an exquisite {ext_color} exterior and a refined {car['interior_color']} interior. It looks absolutely stunning in person. Would you like to schedule a viewing?"
        else:
            features = car['features'][:3] if car['features'] else ["advanced safety tech"]
            feat_str = " and ".join(features)
            return f"Yes, we have a {name} listed in our inventory for {price:,} Riyals. It comes equipped with premium features like {feat_str}. It is in pristine condition. Would you like to register your interest?"

if __name__ == "__main__":
    agent = LLMAgent()
    mock_retrieved = [
        {
            "id": "gv80-2026-3-5t-royal",
            "document": "GV80 3.5T Royal (2026). Price: 335,000 SAR. Body Type: SUV. Engine: 3.5T ROYAL 7P AWD. Exterior: Storr Green. Interior: Smoky Green / Vanilla Beige. Key Features: Panoramic Sunroof, Premium Audio, 360 Camera.",
            "metadata": {
                "name": "GV80 3.5T Royal",
                "price_sar": 335000,
                "exterior_color": "Storr Green",
                "interior_color": "Smoky Green / Vanilla Beige",
                "features": ["Panoramic Sunroof", "Premium Audio", "360 Camera"]
            }
        }
    ]
    
    print("\nTesting Groq LLM streaming with external history...")
    external_history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! I am the Genesis voice assistant. How can I help you today?"},
        {"role": "user", "content": "How much is the GV80?"}
    ]
    for chunk in agent.generate_response_stream("How much is the GV80?", mock_retrieved, history=external_history):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print()
