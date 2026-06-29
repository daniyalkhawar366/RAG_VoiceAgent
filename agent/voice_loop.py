import os
import sys
import re
import time
import asyncio
import threading
import edge_tts

# Add project root to path to ensure clean imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import local components
from rag.retriever import Retriever
from agent.llm import LLMAgent
from agent.audio import AudioHandler
from agent.transcriber import SpeechTranscriber
from agent.session import ConversationSession

# Import central config
from config import (
    TTS_VOICE, TTS_RATE,
    AUDIO_THRESHOLD, AUDIO_SILENCE_DURATION,
    RAG_TOP_K, TEMP_DIR
)

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

audio_handler = None

# Voice configuration settings (driven by config.py)
VOICE_NAME = TTS_VOICE
VOICE_RATE = TTS_RATE

async def main():
    # Ask user if they want to enable barge-in
    print("=" * 60)
    print("Would you like to enable barge-in (interrupt by speaking)? (y/n): ", end="", flush=True)
    barge_in_choice = input().strip().lower()
    barge_in_enabled = barge_in_choice in ["y", "yes", ""]
    print("Barge-in is", "ENABLED" if barge_in_enabled else "DISABLED", "for this session.")
    print("=" * 60)
    print("=" * 60)
    print("       GENESIS CPO RAG VOICE CUSTOMER SUPPORT AGENT")
    print("=" * 60)

    # 1. Initialize RAG Retriever
    print("\n[System] Loading vector database...")
    try:
        retriever = Retriever()
        print("[System] ChromaDB vector store loaded successfully.")
    except Exception as e:
        print(f"[System] Error loading Retriever: {e}")
        print("[System] Make sure you have run 'python rag/indexer.py' first.")
        return

    # 2. Initialize LLM Agent
    print("[System] Initializing LLM agent...")
    llm_agent = LLMAgent()

    # 3. Initialize Audio Handler
    global audio_handler
    print("[System] Initializing audio interface (sounddevice)...")
    audio_handler = AudioHandler(threshold=AUDIO_THRESHOLD, silence_duration=AUDIO_SILENCE_DURATION)
    if barge_in_enabled:
        audio_handler.start_stream()

    # Temporary files
    temp_dir = TEMP_DIR
    os.makedirs(temp_dir, exist_ok=True)
    record_file = os.path.join(temp_dir, "customer_input.wav")
    temp_files = [
        os.path.join(temp_dir, "stream_response_1.mp3"),
        os.path.join(temp_dir, "stream_response_2.mp3")
    ]

    # Initialize ConversationSession
    session = ConversationSession()

    barge_in_triggered = False
    interruption_buffer = None

    async def play_tts_with_barge_in(text_or_file, is_file=False):
        nonlocal barge_in_triggered, interruption_buffer
        
        if is_file:
            temp_file = text_or_file
        else:
            temp_file = os.path.join(temp_dir, "tts_temp_speak.mp3")
            communicate = edge_tts.Communicate(text_or_file, VOICE_NAME, rate=VOICE_RATE)
            await communicate.save(temp_file)
            
        if not barge_in_enabled:
            playback_task = asyncio.create_task(asyncio.to_thread(audio_handler.play_mp3_mci, temp_file))
            await playback_task
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return False

        stop_event = asyncio.Event()
        speak_detected_event = asyncio.Event()
        
        playback_task = asyncio.create_task(asyncio.to_thread(audio_handler.play_mp3_mci, temp_file))
        barge_in_task = asyncio.create_task(audio_handler.listen_for_barge_in(stop_event, speak_detected_event))
        
        while not playback_task.done() and not speak_detected_event.is_set():
            await asyncio.sleep(0.02)
            
        stop_event.set()
        
        if speak_detected_event.is_set():
            barge_in_triggered = True
            playback_task.cancel()
            interruption_buffer = await barge_in_task
            print("\n[Agent] (Interrupted)")
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return True # Interrupted
        else:
            barge_in_task.cancel()
            try:
                await barge_in_task
            except (Exception, asyncio.CancelledError):
                pass
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return False # Finished naturally

    # 4. Greeting
    greeting = (
        "Hello! Thank you for calling Genesis CPO. "
        "I am Alex, your virtual assistant. How can I help you today?"
    )
    print(f"\n[Agent] {greeting}")
    print("[Agent] Synthesizing speech...")
    await play_tts_with_barge_in(greeting)

    # 5. Conversation loop
    transcriber = SpeechTranscriber()
    print("\n[Voice Loop Active] Say 'goodbye', 'bye', or 'exit' to end the call.\n")

    silence_count = 0
    while True:
        # If barge_in was triggered, skip the prompt and typed_input, immediately go to voice recording
        if barge_in_triggered:
            typed_input = ""
            is_barge_in_turn = True
        else:
            is_barge_in_turn = False
            # Hybrid input: wait for keyboard input or Enter key
            print("\n" + "-" * 50)
            print("[System] Press ENTER to speak, or type your query below:")
            try:
                typed_input = input("> ").strip()
            except EOFError:
                typed_input = ""
        
        if typed_input:
            customer_query = typed_input
            print(f"\n[Customer] {customer_query} (Typed)")
            silence_count = 0
        else:
            # Voice Mode
            # If we are resuming from barge-in, we pass the interruption_buffer
            if not barge_in_enabled:
                audio_handler.start_stream()
            if is_barge_in_turn:
                success = audio_handler.record_until_silence(record_file, initial_chunks=interruption_buffer)
            else:
                success = audio_handler.record_until_silence(record_file)
            if not barge_in_enabled:
                audio_handler.stop_stream()
                
            # Reset barge-in flags
            barge_in_triggered = False
            interruption_buffer = None
            
            if not success:
                silence_count += 1
                if silence_count == 1:
                    silence_msg = "Are you still there?"
                else:
                    silence_msg = "Feel free to ask about any vehicle whenever you're ready."
                    silence_count = 0
                    
                print(f"[Agent] {silence_msg}")
                await play_tts_with_barge_in(silence_msg)
                continue

            silence_count = 0
            print("[Agent] Transcribing customer audio...")
            customer_query = transcriber.transcribe(record_file)
            if not customer_query:
                print("[Agent] (Apologies, I couldn't hear that clearly. Could you please repeat?)")
                continue

            print(f"\n[Customer] {customer_query} (Spoken)")

        # Call llm_agent.classify_intent
        active_car_name = session.active_car['metadata']['name'] if session.active_car else "none"
        t_intent_start  = time.perf_counter()
        intent = llm_agent.classify_intent(customer_query, active_car_name)
        t_intent        = time.perf_counter() - t_intent_start
        print(f"[Agent] (Intent Classified: {intent} | {t_intent:.2f}s)")

        # Determine whether to execute RAG query based on intent
        # We now query RAG dynamically for all inventory and spec-related questions.
        run_rag = intent in ["INVENTORY_SEARCH", "COMPARISON", "FEATURE_QUESTION", "PRICE_QUESTION"]        # Route based on intent
        retrieved_cars = []
        if intent == "EXIT":
            farewell = (
                "Thank you for contacting Genesis CPO customer support. "
                "Have a wonderful day! Goodbye."
            )
            print(f"\n[Agent] {farewell}")
            communicate = edge_tts.Communicate(farewell, VOICE_NAME, rate=VOICE_RATE)
            await communicate.save(temp_files[0])
            audio_handler.play_mp3_mci(temp_files[0])
            break
            
        elif run_rag:
            print("[Agent] Searching CPO live inventory...")
            rewritten_query = llm_agent.rewrite_query_if_needed(customer_query, session)

            # Extract filters from this turn and MERGE into session (additive).
            # Only overwrite a key if the new query explicitly mentions it —
            # so "my budget is 100k" keeps body_type=SUV from the previous turn.
            new_filters = llm_agent.extract_filters(customer_query, rewritten_query, session)
            for k, v in new_filters.items():
                if v == "CLEAR":
                    print(f"[Session] Clearing filter '{k}' as requested by LLM.")
                    session.active_filters.pop(k, None)
                    if k in ["max_price", "min_price"]:
                        session.reset_price_filters()
                elif v is not None and v != "" and v != 0:
                    session.active_filters[k] = v

            # Carry persistent body/model context into the retriever so that
            # "budget 100k" after "family car" queries GV80s under 100k, not all cars.
            persistent_body  = session.active_filters.get("body_type")
            persistent_model = session.active_filters.get("model_family")
            persistent_sort  = session.active_filters.get("sort_by")

            t_rag_start    = time.perf_counter()
            retrieved_cars = retriever.query(
                rewritten_query,
                n_results=RAG_TOP_K,
                body_filter=persistent_body,
                model_filter=persistent_model,
                sort_by=persistent_sort
            )

            # If strict metadata filters returned 0 results, do a fallback closest-match query.
            # IMPORTANT: keep body_type filter but DROP price filter so we get the closest
            # on-brand car (e.g. cheapest SUV we carry) rather than a random off-body car.
            # The LLM system prompt (RULE H) will still tell the customer it exceeds their budget.
            if not retrieved_cars:
                print("[RAG] 0 exact matches found. Falling back to body-type-only query...")
                retrieved_cars = retriever.query(
                    rewritten_query,
                    n_results=2,
                    body_filter=persistent_body,    # keep body type (SUV, Sedan, etc.)
                    model_filter=persistent_model,  # keep model family if present
                    sort_by=persistent_sort,        # keep sorting preference!
                    ignore_filters=True             # drop the price where-clause
                )
                
            t_rag = time.perf_counter() - t_rag_start
            session.last_retrieved_cars = retrieved_cars


            print(f"\n[RAG] Retrieved {len(retrieved_cars)} result(s) in {t_rag*1000:.0f}ms:")
            print("-" * 40)
            for idx, res in enumerate(retrieved_cars):
                print(f"  ({idx+1}) {res['metadata']['name']} ({res['metadata']['year']})")
                print(f"      Price: {res['metadata']['price_sar']:,} SAR | Color: {res['metadata']['exterior_color']} / {res['metadata']['interior_color']}")
                print(f"      Engine: {res['metadata']['engine']}")
                print(f"      Distance: {res['distance']:.4f}")
            print("-" * 40 + "\n")
            t_rag_start = None  # already captured
            
        elif intent == "OUT_OF_SCOPE":
            # Direct static safe response for out-of-scope/adversarial queries
            response_text = "I'd be happy to help with Genesis vehicles and inventory-related questions."
            print(f"\n[Agent] {response_text}")
            await play_tts_with_barge_in(response_text)
            
            session.add_user(customer_query)
            session.add_assistant(response_text)
            continue
            
        elif intent == "NEEDS_CONTEXT":
            # No RAG, no inventory — LLM asks which car they mean
            retrieved_cars = []
            
        else:
            # AFFIRMATION, SMALLTALK, CONFUSION, SUMMARY, FILTER_PREVIOUS
            # No new RAG query needed. Just use last known inventory context.
            retrieved_cars = session.last_retrieved_cars

        # 5. Call session.add_user(customer_query)
        session.add_user(customer_query)

        # 6. Stream LLM response
        print("[Agent] Formulating response...")
        t_llm_first_token = None
        t_turn_start      = time.perf_counter()

        chunk_queue = asyncio.Queue()
        sentences_queue = asyncio.Queue()

        def stream_thread_worker(loop):
            nonlocal t_llm_first_token
            try:
                active_car_name = session.active_car['metadata']['name'] if session.active_car else "none"
                first_chunk     = True
                for chunk in llm_agent.generate_response_stream(
                    customer_query=customer_query,
                    retrieved_cars=retrieved_cars,
                    history=session.get_history_for_llm(),
                    session_summary=session.summary(),
                    active_car_name=active_car_name,
                    intent=intent
                ):
                    if first_chunk:
                        t_llm_first_token = time.perf_counter()
                        first_chunk = False
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(chunk), loop)
            except Exception as e:
                print(f"Stream thread error: {e}")
            finally:
                asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)

        loop = asyncio.get_event_loop()
        threading.Thread(target=stream_thread_worker, args=(loop,), daemon=True).start()

        full_response_parts = []

        async def sentence_extractor():
            buffer = ""
            while True:
                chunk = await chunk_queue.get()
                if chunk is None:
                    break

                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_response_parts.append(chunk)

                buffer += chunk

                while True:
                    # Split only on sentence endings (. ! ?) to avoid MCI audio startup delays and word clipping.
                    match = re.search(r'(?<!\d)[\.!?](?:\s+|\n|$)', buffer)
                    if match:
                        idx = match.end()
                        sentence = buffer[:idx].strip()
                        buffer = buffer[idx:]
                        if sentence:
                            await sentences_queue.put(sentence)
                    else:
                        break

            if buffer.strip():
                await sentences_queue.put(buffer.strip())
            await sentences_queue.put(None)

        playback_queue = asyncio.Queue()
        spoken_sentences = []

        async def tts_synthesizer():
            file_idx = 0
            while True:
                sentence = await sentences_queue.get()
                if sentence is None:
                    await playback_queue.put(None)
                    break

                # Skip sentences without any speakable characters to prevent edge-tts crashes
                if not any(c.isalnum() for c in sentence):
                    continue

                temp_file = os.path.join(temp_dir, f"tts_sentence_{file_idx}.mp3")
                file_idx += 1

                try:
                    communicate = edge_tts.Communicate(sentence, VOICE_NAME, rate=VOICE_RATE)
                    await communicate.save(temp_file)
                    await playback_queue.put((temp_file, sentence))
                except Exception as e:
                    print(f"\n[TTS Error] Failed to synthesize sentence '{sentence}': {e}")

        async def audio_playback_worker():
            while True:
                item = await playback_queue.get()
                if item is None:
                    break
                temp_file, sentence = item
                await asyncio.to_thread(audio_handler.play_mp3_mci, temp_file)
                spoken_sentences.append(sentence)
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

        print("\n[Agent] ", end="", flush=True)
        
        extractor_task = asyncio.create_task(sentence_extractor())
        synthesizer_task = asyncio.create_task(tts_synthesizer())
        playback_task = asyncio.create_task(audio_playback_worker())
        
        if barge_in_enabled:
            stop_event = asyncio.Event()
            speak_detected_event = asyncio.Event()
            barge_in_task = asyncio.create_task(audio_handler.listen_for_barge_in(stop_event, speak_detected_event))
            
            while not playback_task.done() and not speak_detected_event.is_set():
                await asyncio.sleep(0.02)
                
            stop_event.set()
            
            if speak_detected_event.is_set():
                barge_in_triggered = True
                extractor_task.cancel()
                synthesizer_task.cancel()
                playback_task.cancel()
                
                try:
                    await asyncio.gather(extractor_task, synthesizer_task, playback_task, return_exceptions=True)
                except Exception:
                    pass
                    
                interruption_buffer = await barge_in_task
                print("\n[Agent] (Interrupted)")
            else:
                barge_in_task.cancel()
                try:
                    await barge_in_task
                except (Exception, asyncio.CancelledError):
                    pass
                await playback_task
        else:
            await asyncio.gather(extractor_task, synthesizer_task, playback_task)
            
        # Clean up any leftover sentence files that were synthesized but never played
        for f in os.listdir(temp_dir):
            if f.startswith("tts_sentence_") and f.endswith(".mp3"):
                try:
                    os.remove(os.path.join(temp_dir, f))
                except Exception:
                    pass
        print()

        # 8. Save assistant response to session (only what was actually spoken)
        full_response_text = " ".join(spoken_sentences).strip()
        if not full_response_text:
            # Fallback if interrupted before even 1 sentence was spoken
            full_response_text = "".join(full_response_parts).strip()
        session.add_assistant(full_response_text)

        # --- Per-Turn Latency Summary ---
        t_total   = time.perf_counter() - t_turn_start
        llm_first = (t_llm_first_token - t_turn_start) if t_llm_first_token else 0.0
        print(
            f"\n[Perf] Intent: {t_intent:.2f}s | "
            f"LLM first token: {llm_first:.2f}s | "
            f"Turn total: {t_total:.2f}s"
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[System] Call disconnected by user. Goodbye!")
    finally:
        if audio_handler:
            audio_handler.stop_stream()