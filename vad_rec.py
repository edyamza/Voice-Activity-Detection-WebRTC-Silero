import webrtcvad
import numpy as np
import wave
import simpleaudio as sa
import time
from collections import deque

vad = webrtcvad.Vad(3)

AMPLITUDE_THRESHOLD = 200
VARIATIE_MINIMA = 30
WINDOW_CONSECUTIVE = 3

def classify_audio(frame, rate, history, prev_amplitude, consecutive_speech):
    # Validare lungime frame
    if len(frame) not in [320, 640, 960]:  # corespunzător 10ms, 20ms, 30ms la 16kHz
        print(f"[Frame Skipped] Dimensiune invalidă: {len(frame)} bytes")
        return "TĂCERE"

    audio_data = np.frombuffer(frame, dtype=np.int16)
    amplitude = np.abs(audio_data).mean()
    history.append(amplitude)

    if len(history) > 10:
        history.popleft()

    amplitude_variation = np.std(history)

    try:
        is_speech = vad.is_speech(frame, rate)
    except webrtcvad.VadError as e:
        print(f"[VAD Error] {e}")
        return "TĂCERE"

    if is_speech and amplitude > AMPLITUDE_THRESHOLD and amplitude_variation > VARIATIE_MINIMA:
        consecutive_speech.append(1)
    else:
        consecutive_speech.append(0)

    if len(consecutive_speech) > WINDOW_CONSECUTIVE:
        consecutive_speech.popleft()

    return "VORBIRE" if sum(consecutive_speech) == WINDOW_CONSECUTIVE else "TĂCERE"


def process_audio_file_gui(file_path, update_signal, progress_signal, stop_processing):
    results = []
    history = deque()
    consecutive_speech = deque()
    prev_amplitude = 0

    with wave.open(file_path, 'rb') as wf:
        rate = wf.getframerate()
        total_frames = wf.getnframes()

        # Frame de 30ms → 480 sample x 2 bytes/sample = 960 bytes
        frame_duration_ms = 30
        frame_size = int(rate * (frame_duration_ms / 1000.0))  # in samples

        while True:
            if stop_processing:
                break

            frame = wf.readframes(frame_size)
            if len(frame) < frame_size * 2:  # 2 bytes per sample (16-bit audio)
                break

            result = classify_audio(frame, rate, history, prev_amplitude, consecutive_speech)
            results.append(result)

            update_signal.emit(result)
            progress = int((wf.tell() / total_frames) * 100)
            progress_signal.emit(progress)

            prev_amplitude = np.abs(np.frombuffer(frame, dtype=np.int16)).mean()
            time.sleep(frame_duration_ms / 1000.0)

    return results


def process_audio_file_silent(file_path, stop_processing=False):
    from types import SimpleNamespace

    dummy_signal = SimpleNamespace()
    dummy_signal.emit = lambda x: None

    return process_audio_file_gui(file_path, dummy_signal, dummy_signal, stop_processing)
