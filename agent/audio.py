import os
import sys
import time
import ctypes
import platform
import subprocess
import asyncio
import queue
from collections import deque
import numpy as np
import sounddevice as sd
import soundfile as sf

# Reconfigure stdout for UTF-8 compatibility on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import central config with safe fallbacks for standalone use
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
try:
    from config import (
        AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE,
        AUDIO_THRESHOLD, AUDIO_SILENCE_DURATION,
        AUDIO_NO_SPEECH_TIMEOUT, AUDIO_NOISE_MARGIN, AUDIO_NOISE_CAP,
    )
except ImportError:
    AUDIO_SAMPLE_RATE       = 16000
    AUDIO_CHUNK_SIZE        = 1024
    AUDIO_THRESHOLD         = 0.02
    AUDIO_SILENCE_DURATION  = 1.4
    AUDIO_NO_SPEECH_TIMEOUT = 7.0
    AUDIO_NOISE_MARGIN      = 1.8
    AUDIO_NOISE_CAP         = 0.08

class AudioHandler:
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, threshold=AUDIO_THRESHOLD, silence_duration=AUDIO_SILENCE_DURATION):
        """
        threshold: RMS amplitude threshold below which is considered silence.
        silence_duration: seconds of silence before we stop recording.
        """
        self.sample_rate      = sample_rate
        self.threshold        = threshold
        self.silence_duration = silence_duration
        self.chunk_size       = AUDIO_CHUNK_SIZE
        self.current_process  = None
        self.is_playing       = False
        self.audio_queue      = queue.Queue()
        self.stream           = None

    def start_stream(self):
        """
        Starts the persistent InputStream.
        """
        def callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy())
            
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.chunk_size,
                callback=callback
            )
            self.stream.start()
            print("[Audio] Persistent input stream started successfully.")
        except Exception as e:
            print(f"[Audio Error] Failed to start persistent input stream: {e}")

    def stop_stream(self):
        """
        Stops and closes the persistent stream.
        """
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def record_until_silence(self, output_filepath, initial_chunks=None):
        """
        Records from the persistent audio queue until user stops speaking (silence detected).
        Adapts dynamically to the ambient noise floor in the room (VAD enhancement).
        Saves the recorded audio as a WAV file.
        """
        print("\n[Agent] Listening... (start speaking)")
        
        audio_data = []
        speaking_started = False
        silence_start_time = None
        
        # Clear the queue first to discard any old audio before this turn started
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break
        
        # If we have initial chunks, we prepopulate audio_data and skip VAD calibration
        if initial_chunks:
            audio_data = list(initial_chunks)
            speaking_started = True
            dynamic_threshold = self.threshold
            print(f"[Audio] Resuming recording from barge-in buffer ({len(audio_data)} chunks).")
            
        if not initial_chunks:
            # 1. Calibrate to dynamic noise floor in first 300ms using the queue
            warmup_chunks = max(3, int(0.3 * self.sample_rate / self.chunk_size))
            noise_rms_list = []
            
            count = 0
            while count < warmup_chunks:
                try:
                    # Timeout to avoid blocking forever if stream failed
                    chunk = self.audio_queue.get(timeout=1.0)
                    audio_data.append(chunk)
                    noise_rms_list.append(np.sqrt(np.mean(chunk**2)))
                    count += 1
                except Exception as e:
                    print(f"[Audio Error] Timeout reading warmup chunks from persistent stream: {e}")
                    return False
                    
            ambient_noise = np.mean(noise_rms_list)
            # Set threshold dynamically: ambient noise multiplied by safety margin
            dynamic_threshold = max(self.threshold, ambient_noise * AUDIO_NOISE_MARGIN)
            # Cap threshold to a reasonable maximum to avoid blocking speech in noisy rooms
            dynamic_threshold = min(dynamic_threshold, AUDIO_NOISE_CAP)
            
            print(f"[Audio] Noise floor calibrated: {ambient_noise:.4f}. Dynamic threshold: {dynamic_threshold:.4f}")
        else:
            # Set dynamic threshold to a baseline if we skipped calibration
            dynamic_threshold = self.threshold
            
        # 2. Main recording loop
        while True:
            try:
                chunk = self.audio_queue.get(timeout=2.0)
            except Exception as e:
                print(f"[Audio Error] Timeout waiting for audio chunk: {e}")
                break
                
            audio_data.append(chunk)
            
            # Calculate Root Mean Square (RMS) amplitude of the chunk
            rms = np.sqrt(np.mean(chunk**2))
            
            if not speaking_started:
                if rms > dynamic_threshold:
                    print("[Agent] Speaking detected...")
                    speaking_started = True
            else:
                if rms < dynamic_threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > self.silence_duration:
                        print("[Agent] Silence detected. Stopping recording.")
                        break
                else:
                    # Reset silence timer when user speaks
                    silence_start_time = None
                    
            # Safety timeout: if user doesn't start speaking, stop
            if not speaking_started and len(audio_data) * self.chunk_size / self.sample_rate > AUDIO_NO_SPEECH_TIMEOUT:
                print("[Agent] No speech detected. Stopping.")
                return False
                
        # Concatenate all recorded chunks
        recording = np.concatenate(audio_data, axis=0)
        
        # Save as WAV file
        os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
        sf.write(output_filepath, recording, self.sample_rate)
        return True

    def play_wav(self, filepath):
        """
        Plays back a WAV file.
        """
        if not os.path.exists(filepath):
            print(f"Error: Audio file {filepath} not found.")
            return
            
        data, fs = sf.read(filepath)
        sd.play(data, fs)
        sd.wait()

    def play_mp3_mci(self, filepath):
        """
        Plays back an MP3 file.
        Uses cross-platform memory-based sounddevice playback as the primary low-latency driver.
        Falls back to legacy Windows MCI, macOS afplay, or Linux mpg123 if sounddevice fails.
        Can be interrupted by setting self.is_playing = False.
        """
        if not os.path.exists(filepath):
            print(f"Error: Audio file {filepath} not found.")
            return False
            
        abs_path = os.path.abspath(filepath)
        sys_platform = platform.system().lower()
        self.is_playing = True
        
        # Primary low-latency cross-platform driver: sounddevice + soundfile
        try:
            data, fs = sf.read(abs_path)
            sd.play(data, fs)
            
            duration = len(data) / fs
            start_time = time.time()
            while self.is_playing:
                if time.time() - start_time >= duration:
                    break
                time.sleep(0.02)
                
            if not self.is_playing:
                sd.stop()
            self.is_playing = False
            return True
        except Exception as e:
            print(f"[Audio Warning] Native sounddevice playback failed: {e}. Falling back to OS driver...")

        # Fallback 1: Windows native MCI (winmm.dll)
        if "windows" in sys_platform:
            try:
                # Close any previous instance first
                ctypes.windll.winmm.mciSendStringW("close my_tts", None, 0, 0)
                
                # Open file
                open_cmd = f'open "{abs_path}" type mpegvideo alias my_tts'
                res = ctypes.windll.winmm.mciSendStringW(open_cmd, None, 0, 0)
                if res != 0:
                    self.is_playing = False
                    return False
                    
                # Short delay to allow sound card driver initialization
                time.sleep(0.15)
                    
                # Play file
                res = ctypes.windll.winmm.mciSendStringW("play my_tts", None, 0, 0)
                if res != 0:
                    self.is_playing = False
                    return False
                    
                # Poll mode status
                status = ctypes.create_unicode_buffer(250)
                while self.is_playing:
                    ctypes.windll.winmm.mciSendStringW("status my_tts mode", status, 250, 0)
                    mode = status.value.strip()
                    if mode != "playing":
                        break
                    time.sleep(0.05)
                    
                # Close alias
                ctypes.windll.winmm.mciSendStringW("close my_tts", None, 0, 0)
                self.is_playing = False
                return True
            except Exception as e:
                print(f"MCI Playback failed: {e}")
                self.is_playing = False
                return False
                
        # Fallback 2: macOS Native Playback (afplay)
        elif "darwin" in sys_platform:
            try:
                self.current_process = subprocess.Popen(["afplay", abs_path])
                self.current_process.wait()
                self.current_process = None
                self.is_playing = False
                return True
            except Exception as e:
                print(f"afplay playback failed: {e}")
                self.is_playing = False
                return False
                
        # Fallback 3: Linux / Unix fallback players
        else:
            for player in ["mpg123", "play", "ffplay"]:
                try:
                    if player == "ffplay":
                        cmd = ["ffplay", "-nodisp", "-autoexit", abs_path]
                    else:
                        cmd = [player, abs_path]
                    self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    self.current_process = None
                    self.is_playing = False
                    return True
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
            print("Warning: No compatible CLI MP3 player found on this Linux system. Please install mpg123.")
            self.is_playing = False
            return False

    def stop_playback(self):
        """
        Immediately stops any active audio playback.
        """
        self.is_playing = False
        try:
            sd.stop()
        except Exception:
            pass
            
        sys_platform = platform.system().lower()
        if "windows" in sys_platform:
            try:
                ctypes.windll.winmm.mciSendStringW("stop my_tts", None, 0, 0)
                ctypes.windll.winmm.mciSendStringW("close my_tts", None, 0, 0)
            except Exception as e:
                print(f"[Audio] Failed to stop Windows MCI playback: {e}")
        else:
            if self.current_process:
                try:
                    self.current_process.terminate()
                except Exception:
                    pass
                self.current_process = None

    async def listen_for_barge_in(self, stop_event, speak_detected_event):
        """
        Monitors the persistent audio queue for barge-in speech concurrently with playback.
        Stores a sliding window buffer of chunks so that the beginning of their
        interruption speech is not truncated when recording resumes.
        
        Returns:
            list of numpy arrays (buffered chunks) if interrupted, else None.
        """
        barge_threshold = max(0.008, self.threshold * 0.8)
        rolling_buffer = deque(maxlen=12)
        
        # Clear the queue of old chunks before playback started
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except Exception:
                break
                
        # Read from persistent queue asynchronously
        consecutive_speech_chunks = 0
        required_speech_chunks = 3  # 3 chunks * 64ms = ~192ms of continuous speech to trigger barge-in
        
        while not stop_event.is_set() and not speak_detected_event.is_set():
            while not self.audio_queue.empty():
                try:
                    chunk = self.audio_queue.get_nowait()
                    rolling_buffer.append(chunk)
                    rms = np.sqrt(np.mean(chunk**2))
                    if rms > barge_threshold:
                        consecutive_speech_chunks += 1
                        if consecutive_speech_chunks >= required_speech_chunks:
                            speak_detected_event.set()
                            break
                    else:
                        consecutive_speech_chunks = 0
                except Exception:
                    pass
            await asyncio.sleep(0.02)
            
        if speak_detected_event.is_set():
            # Stop playback immediately!
            self.stop_playback()
            return list(rolling_buffer)
            
        return None

if __name__ == "__main__":
    handler   = AudioHandler(threshold=AUDIO_THRESHOLD, silence_duration=1.2)
    # Use a path relative to this file so it works on any machine
    test_file = os.path.join(_BASE_DIR, "data", "test_record.wav")
    
    print("Testing recording. Please speak into your microphone.")
    success = handler.record_until_silence(test_file)
    if success:
        print("Recording saved. Playing it back...")
        handler.play_wav(test_file)
        print("Playback finished.")
