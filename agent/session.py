import os
import sys

# Resolve project root so config is importable regardless of CWD
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

try:
    from config import RAG_FUZZY_MATCH_SCORE, RAG_HISTORY_WINDOW
except ImportError:
    RAG_FUZZY_MATCH_SCORE = 70
    RAG_HISTORY_WINDOW    = 20


class ConversationSession:
    def __init__(self):
        self.history: list[dict] = []          # full role/content message history
        self.last_retrieved_cars: list = []     # cars returned by last RAG query
        self.active_car: dict | None = None     # specific car currently being discussed
        self.active_filters: dict = {}          # body_type, model family, price range context
        self.turn: int = 0

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})
        self.turn += 1

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self.active_car = self._extract_mentioned_car(text)

    def _extract_mentioned_car(self, text: str) -> dict | None:
        from rapidfuzz import fuzz
        best_match = None
        best_score = 0
        for car in self.last_retrieved_cars:
            name  = car['metadata']['name']
            score = fuzz.partial_ratio(name.lower(), text.lower())
            if score > RAG_FUZZY_MATCH_SCORE and score > best_score:
                best_score = score
                best_match = car
        return best_match if best_match else self.active_car

    def get_history_for_llm(self, max_turns: int = RAG_HISTORY_WINDOW) -> list[dict]:
        return self.history[-(max_turns * 2):]

    def summary(self) -> str:
        parts = []
        if self.active_car:
            m        = self.active_car['metadata']
            features = ', '.join(m.get('features', []))
            parts.append(
                f"Currently discussing: {m['name']} | "
                f"{m['price_sar']:,} SAR | "
                f"Exterior: {m['exterior_color']} | "
                f"Interior: {m['interior_color']} | "
                f"Engine: {m['engine']} | "
                f"Body: {m.get('body_type', 'unknown')} | "
                f"Fuel: {m.get('fuel_type', 'unknown')} | "
                f"Features: {features}"
            )
        if self.active_filters:
            parts.append(f"Active search filters: {self.active_filters}")
            # Explicitly surface the customer's stated budget so the LLM always
            # knows the price ceiling/floor even when the RAG fallback drops the
            # price where-clause to fetch the closest available match.
            max_p = self.active_filters.get("max_price")
            min_p = self.active_filters.get("min_price")
            if max_p:
                try:
                    max_p_int = int(max_p)
                    formatted_max = f"{max_p_int:,}"
                except ValueError:
                    formatted_max = str(max_p)
                parts.append(
                    f"CUSTOMER BUDGET CONSTRAINT: Customer stated a maximum budget of "
                    f"{formatted_max} SAR. Do NOT claim any car priced above this fits their budget."
                )
            if min_p:
                try:
                    min_p_int = int(min_p)
                    formatted_min = f"{min_p_int:,}"
                except ValueError:
                    formatted_min = str(min_p)
                parts.append(
                    f"CUSTOMER MINIMUM PRICE: Customer wants vehicles priced at least {formatted_min} SAR."
                )
        return "\n".join(parts) if parts else "No specific vehicle in focus yet."

    def reset_price_filters(self):
        """Clear stale price constraints when the customer starts a fresh search topic."""
        self.active_filters.pop("max_price", None)
        self.active_filters.pop("min_price", None)
