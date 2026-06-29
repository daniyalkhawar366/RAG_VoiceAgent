import os
import re
import sys
import chromadb
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()



# Import central config — fall back to safe defaults if imported standalone
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    import sys as _sys
    _sys.path.insert(0, BASE_DIR)
    from config import (
        RAG_TOP_K,
        RAG_MODEL_FILTER_LIMIT,
        RAG_PRICE_RANGE_MARGIN,
        RAG_SIMILARITY_THRESHOLD,
        DB_PATH
    )
except ImportError:
    RAG_TOP_K = 5
    RAG_MODEL_FILTER_LIMIT = 25
    RAG_PRICE_RANGE_MARGIN = 0.15
    RAG_SIMILARITY_THRESHOLD = 0.85

# Color keywords list removed, color re-ranking is now handled by the LLM system prompt.

def parse_query_filters(query_text, body_filter=None):
    """
    Parses a natural language query to extract structured metadata filters for ChromaDB.
    Returns:
        where_clause (dict or None): ChromaDB metadata filter dictionary.
        body_type (str or None): Extracted body type filter.
    """
    query_lower = query_text.lower()
    
    # 1. Detect body type
    body_type = None
    if "suv" in query_lower:
        body_type = "SUV"
    elif "sedan" in query_lower:
        body_type = "Sedan"
    elif "coupe" in query_lower:
        body_type = "Coupe"
        
    # If not explicitly mentioned, carry over the body style context
    if not body_type and body_filter:
        body_type = body_filter
        
    # Helper to convert price strings
    def text_to_val(num_str, suffix):
        clean_num = re.sub(r'[^\d]', '', num_str)
        if not clean_num:
            return 0
        val = int(clean_num)
        if "k" in suffix or "thousand" in suffix or val < 1000:
            val *= 1000
        return val

    # 2. Match price patterns
    between_match = re.search(r'(?:between|from)\s+([\d,]+)\s*(k|thousand)?\s*(?:and|to)\s+([\d,]+)\s*(k|thousand)?', query_lower)
    under_match = re.search(r'(?:under|less than|below|cheaper than|<)\s*([\d,]+)\s*(k|thousand)?', query_lower)
    over_match = re.search(r'(?:over|more than|above|greater than|>)\s*([\d,]+)\s*(k|thousand)?', query_lower)
    around_match = re.search(r'(?:around|about|approx|approximately)\s*([\d,]+)\s*(k|thousand)?', query_lower)

    conditions = []
    
    # Add body type metadata filter
    if body_type:
        conditions.append({"body_type": {"$eq": body_type}})
        
    # Add price constraints
    if between_match:
        val1 = text_to_val(between_match.group(1), between_match.group(2) or "")
        val2 = text_to_val(between_match.group(3), between_match.group(4) or "")
        low, high = min(val1, val2), max(val1, val2)
        conditions.append({"price_sar": {"$gte": low}})
        conditions.append({"price_sar": {"$lte": high}})
    elif under_match:
        val = text_to_val(under_match.group(1), under_match.group(2) or "")
        conditions.append({"price_sar": {"$lte": val}})
    elif over_match:
        val = text_to_val(over_match.group(1), over_match.group(2) or "")
        conditions.append({"price_sar": {"$gte": val}})
    elif around_match:
        val = text_to_val(around_match.group(1), around_match.group(2) or "")
        low  = int(val * (1 - RAG_PRICE_RANGE_MARGIN))
        high = int(val * (1 + RAG_PRICE_RANGE_MARGIN))
        conditions.append({"price_sar": {"$gte": low}})
        conditions.append({"price_sar": {"$lte": high}})
        
    # Construct final ChromaDB where clause
    where_clause = None
    if len(conditions) > 1:
        where_clause = {"$and": conditions}
    elif len(conditions) == 1:
        where_clause = conditions[0]
        
    return where_clause, body_type

class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name="car_inventory")
        
    def query(self, query_text, n_results=3, model_filter=None, body_filter=None, sort_by=None, ignore_filters=False):
        """
        Queries the ChromaDB vector database using metadata-aware filters, hybrid sorting,
        model post-filtering, and post-retrieval color reranking.
        """
        query_lower = query_text.lower()
        where_clause, body_type = parse_query_filters(query_text, body_filter=body_filter)
        
        if ignore_filters:
            where_clause = None
        
        # Determine sorting preference driven entirely by LLM context
        is_cheapest = (sort_by == "price_asc")
        is_expensive = (sort_by == "price_desc")
        
        # Target model is provided strictly by the LLM filter context
        target_model = model_filter
        if isinstance(target_model, str) and target_model.lower().strip() == "genesis":
            target_model = None
        
        if where_clause:
            print(f"[RAG Info] Applied Metadata Filter: {where_clause}")
            
        if is_cheapest or is_expensive:
            # Query-filtered programmatic sort
            if where_clause:
                results = self.collection.get(where=where_clause, limit=100)
            else:
                results = self.collection.get(limit=100)
                
            matches = []
            if results['documents']:
                for doc, meta, doc_id in zip(
                    results['documents'], 
                    results['metadatas'], 
                    results['ids']
                ):
                    features_list = [f.strip() for f in meta.get("features_str", "").split(",") if f.strip()]
                    matches.append({
                        "id": doc_id,
                        "document": doc,
                        "metadata": {
                            "url": meta.get("url", ""),
                            "name": meta.get("name", ""),
                            "price_sar": meta.get("price_sar", 0),
                            "year": meta.get("year", 0),
                            "body_type": meta.get("body_type", ""),
                            "exterior_color": meta.get("exterior_color", ""),
                            "interior_color": meta.get("interior_color", ""),
                            "fuel_type": meta.get("fuel_type", ""),
                            "transmission": meta.get("transmission", ""),
                            "engine": meta.get("engine", ""),
                            "features": features_list
                        },
                        "distance": 0.0
                    })
                    
                valid_matches = [m for m in matches if m["metadata"]["price_sar"] > 0]
                
                # Apply model filter
                if target_model:
                    if isinstance(target_model, list):
                        # Group by model and find the cheapest/most expensive for EACH model
                        model_matches = []
                        for t in target_model:
                            t_matches = [m for m in valid_matches if str(t).lower() in m["metadata"]["name"].lower()]
                            if is_cheapest:
                                t_matches.sort(key=lambda x: x["metadata"]["price_sar"])
                            else:
                                t_matches.sort(key=lambda x: x["metadata"]["price_sar"], reverse=True)
                            model_matches.append(t_matches)
                        
                        # Interleave to ensure diverse representation
                        interleaved = []
                        max_len = max((len(l) for l in model_matches), default=0)
                        for i in range(max_len):
                            for t_list in model_matches:
                                if i < len(t_list):
                                    if t_list[i] not in interleaved:
                                        interleaved.append(t_list[i])
                        return interleaved[:n_results]
                    else:
                        valid_matches = [m for m in valid_matches if str(target_model).lower() in m["metadata"]["name"].lower()]
                
                if is_cheapest:
                    valid_matches.sort(key=lambda x: x["metadata"]["price_sar"])
                else:
                    valid_matches.sort(key=lambda x: x["metadata"]["price_sar"], reverse=True)
                return valid_matches[:n_results]

        # Query a wider pool of results if model filter is active
        query_limit = 100 if target_model else n_results
        
        if where_clause:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=query_limit,
                where=where_clause
            )
        else:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=query_limit
            )
        
        matches = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, dist, doc_id in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0],
                results['ids'][0]
            ):
                features_list = [f.strip() for f in meta.get("features_str", "").split(",") if f.strip()]
                
                matches.append({
                    "id": doc_id,
                    "document": doc,
                    "metadata": {
                        "url": meta.get("url", ""),
                        "name": meta.get("name", ""),
                        "price_sar": meta.get("price_sar", 0),
                        "year": meta.get("year", 0),
                        "body_type": meta.get("body_type", ""),
                        "exterior_color": meta.get("exterior_color", ""),
                        "interior_color": meta.get("interior_color", ""),
                        "fuel_type": meta.get("fuel_type", ""),
                        "transmission": meta.get("transmission", ""),
                        "engine": meta.get("engine", ""),
                        "features": features_list
                    },
                    "distance": dist
                })
        
        # Apply model filter with diverse representation
        if target_model and matches:
            if isinstance(target_model, list):
                # Interleave results from each target model to ensure representation regardless of semantic skew
                model_matches = []
                for t in target_model:
                    t_matches = [m for m in matches if str(t).lower() in m["metadata"]["name"].lower()]
                    model_matches.append(t_matches)
                
                interleaved = []
                max_len = max((len(l) for l in model_matches), default=0)
                for i in range(max_len):
                    for t_list in model_matches:
                        if i < len(t_list):
                            if t_list[i] not in interleaved:
                                interleaved.append(t_list[i])
                matches = interleaved
            else:
                matches = [m for m in matches if str(target_model).lower() in m["metadata"]["name"].lower()]

        # --- Similarity threshold gate ---
        # ONLY apply when no metadata where_clause was used.
        #
        # When a price/body-type filter is active, the rewritten query text is a
        # normalised string like "price between 200000 and 210000" which is
        # semantically distant from car descriptions even for perfectly correct
        # results. Gating on distance in that case silently drops valid matches.
        #
        # For pure semantic searches (no filter), the gate correctly rejects
        # off-brand / out-of-inventory queries like "do you have a Ferrari?"
        if not where_clause:
            before_gate = len(matches)
            matches = [m for m in matches if m["distance"] <= RAG_SIMILARITY_THRESHOLD]
            if len(matches) < before_gate:
                print(f"[RAG] Similarity gate removed {before_gate - len(matches)} weak match(es) "
                      f"(threshold: {RAG_SIMILARITY_THRESHOLD}).")
        else:
            print(f"[RAG] Metadata filter active — similarity gate bypassed ({len(matches)} result(s) kept).")

        return matches[:n_results]


if __name__ == "__main__":
    retriever = Retriever()
    res = retriever.query("cheapest one", model_filter="GV80", body_filter="SUV")
    print(f"Cheapest GV80 SUV match: {res[0]['metadata']['name']} - {res[0]['metadata']['price_sar']:,} SAR")
