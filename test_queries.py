import os
import sys
import re
from colorama import init, Fore, Style

# Add project root to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Initialize colorama for colored terminal output
init(autoreset=True)

try:
    from rag.retriever import Retriever
    from agent.llm import LLMAgent
except ImportError as e:
    print(Fore.RED + f"Import Error: {e}")
    print("Make sure you are running the script from the project root directory.")
    sys.exit(1)

# Initialize systems
print(Fore.CYAN + "Initializing test environment...")
retriever = Retriever()
llm_agent = LLMAgent()

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
# NOTE: Assertions are written to be INVENTORY-AGNOSTIC wherever possible.
# They verify structural / ordering properties rather than hardcoded SAR values
# so they remain green even if listing prices change after a re-scrape.
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "category": "Price & Numeric Queries",
        "query": "What is your cheapest car?",
        # The first result must have the lowest price of all returned matches.
        "assertion": lambda matches: (
            len(matches) > 0 and
            matches[0]["metadata"]["price_sar"] == min(m["metadata"]["price_sar"] for m in matches)
        ),
        "expected": "Results sorted ascending by price; first match is the cheapest in inventory"
    },
    {
        "category": "Price & Numeric Queries",
        "query": "What is your most expensive car?",
        # The first result must have the highest price of all returned matches.
        "assertion": lambda matches: (
            len(matches) > 0 and
            matches[0]["metadata"]["price_sar"] == max(m["metadata"]["price_sar"] for m in matches)
        ),
        "expected": "Results sorted descending by price; first match is the most expensive in inventory"
    },
    {
        "category": "Price & Numeric Queries",
        "query": "Do you have anything under 150,000 SAR?",
        # Every returned result must respect the price ceiling.
        "assertion": lambda matches: (
            len(matches) > 0 and
            all(m["metadata"]["price_sar"] <= 150000 for m in matches)
        ),
        "expected": "All returned cars priced <= 150,000 SAR"
    },
    {
        "category": "Price & Numeric Queries",
        "query": "What can I get between 200,000 and 210,000 SAR?",
        # Every returned result must fall in the stated range.
        "assertion": lambda matches: (
            len(matches) > 0 and
            all(200000 <= m["metadata"]["price_sar"] <= 210000 for m in matches)
        ),
        "expected": "All returned cars priced between 200k and 210k SAR"
    },
    {
        "category": "Price & Numeric Queries",
        "query": "Show me cars above 300,000 SAR",
        # Every returned result must exceed the floor price.
        "assertion": lambda matches: (
            len(matches) > 0 and
            all(m["metadata"]["price_sar"] >= 300000 for m in matches)
        ),
        "expected": "All returned cars priced >= 300,000 SAR"
    },
    {
        "category": "Color-Specific Queries",
        "query": "Do you have anything in Capri Blue?",
        # At least one result must contain 'capri blue' in its exterior color.
        "assertion": lambda matches: (
            len(matches) > 0 and
            any("capri blue" in m["metadata"]["exterior_color"].lower() for m in matches)
        ),
        "expected": "At least one match with exterior color containing 'Capri Blue'"
    },
    {
        "category": "Color-Specific Queries",
        "query": "I want a white SUV",
        # All returned results must be SUVs (body type filter applied).
        "assertion": lambda matches: (
            len(matches) > 0 and
            all(m["metadata"]["body_type"] == "SUV" for m in matches)
        ),
        "expected": "All returned matches are SUVs (white color preference handled by LLM)"
    },
    {
        "category": "Color-Specific Queries",
        "query": "Do you have anything in green?",
        # At least one result must have 'green' somewhere in its exterior color name.
        "assertion": lambda matches: (
            len(matches) > 0 and
            any("green" in m["metadata"]["exterior_color"].lower() for m in matches)
        ),
        "expected": "At least one match with a green exterior color (any shade)"
    },
    {
        "category": "Adversarial / Hallucination Guard",
        "query": "Do you have a Ferrari?",
        # The similarity gate should return zero matches for a brand we don't carry.
        # If a match IS returned, its distance must be non-zero (vector search only).
        # Primary check: no result should mention 'Ferrari' in the name.
        "assertion": lambda matches: (
            all("ferrari" not in m["metadata"]["name"].lower() for m in matches)
        ),
        "expected": "No Ferrari matches returned (not in inventory)"
    },
    {
        "category": "Adversarial / Hallucination Guard",
        "query": "Which car has 900 horsepower?",
        # No match should exist; if the similarity gate works we get 0 results.
        # Assertion: result list is empty OR none of the documents mention 900 hp.
        "assertion": lambda matches: (
            len(matches) == 0 or
            all("900" not in m["document"] for m in matches)
        ),
        "expected": "Zero results or no document mentions 900 hp"
    },
    {
        "category": "Body Type Filter",
        "query": "Show me all your sedans",
        # All returned results must have body_type == 'Sedan'.
        "assertion": lambda matches: (
            len(matches) > 0 and
            all(m["metadata"]["body_type"] == "Sedan" for m in matches)
        ),
        "expected": "All returned matches are Sedans"
    },
]


def run_tests():
    print("\n" + "=" * 80)
    print("             GENESIS CPO RAG VOICE AGENT EVALUATION RUNNER")
    print("=" * 80 + "\n")

    passed_count = 0
    total_count  = len(TEST_CASES)

    for idx, tc in enumerate(TEST_CASES):
        category  = tc["category"]
        query     = tc["query"]
        expected  = tc["expected"]
        assertion = tc["assertion"]

        print(Fore.BLUE  + f"[{category}] Test Case {idx+1}/{total_count}")
        print(Fore.WHITE + Style.BRIGHT + f'Query: "{query}"')
        print(Fore.YELLOW + f"Expected: {expected}")

        # 1. Run RAG Retrieval
        # We manually mock the LLM's sort extraction here since the test bypasses the LLM
        sort_override = "price_asc" if "cheapest" in query.lower() else "price_desc" if "expensive" in query.lower() else None
        matches = retriever.query(query, n_results=5, sort_by=sort_override)

        print(Fore.CYAN + f"RAG Retrieved {len(matches)} match(es):")
        for rank, m in enumerate(matches):
            meta = m["metadata"]
            dist = m.get("distance", 0.0)
            print(f"  ({rank+1}) {meta['name']} ({meta['year']})  dist={dist:.4f}")
            print(f"      Price: {meta['price_sar']:,} SAR | "
                  f"Color: {meta['exterior_color']} / {meta['interior_color']}")
            print(f"      Body: {meta['body_type']} | Engine: {meta['engine']}")

        # Run assertion
        try:
            assert_status = assertion(matches)
        except Exception:
            assert_status = False

        if assert_status:
            print(Fore.GREEN + Style.BRIGHT + "RETRIEVAL ASSERTION: PASSED ✔\n")
            passed_count += 1
        else:
            print(Fore.RED + Style.BRIGHT + "RETRIEVAL ASSERTION: FAILED ❌\n")

        # 2. Sample LLM Response (non-streaming for clean logs)
        print(Fore.CYAN + "Agent Response:")
        response = llm_agent.generate_response(query, matches)
        print(Fore.GREEN + f'"{response}"')
        print("-" * 80 + "\n")

        # Prevent Groq TPM rate limit during rapid-fire testing
        import time
        time.sleep(3.5)

    print("=" * 80)
    print(f"EVALUATION SUMMARY: {passed_count}/{total_count} Retrieval Tests Passed")
    print("=" * 80)

    if passed_count == total_count:
        print(Fore.GREEN + Style.BRIGHT + "SUCCESS: All retrieval assertions passed! Ready for demonstration.")
    else:
        print(Fore.RED + Style.BRIGHT + "WARNING: Some assertions failed. Review retrieval or similarity threshold.")


if __name__ == "__main__":
    run_tests()
