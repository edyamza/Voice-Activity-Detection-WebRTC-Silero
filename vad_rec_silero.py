import torch
import numpy as np
import wave
import simpleaudio as sa
import time
import sys

sys.path.append('/Users/daemondog/Desktop/proiect PSV - VAD/silero-0.4.1')

try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils
    print("Modelul Silero VAD a fost încărcat cu succes.")
except Exception as e:
    raise RuntimeError(f"Eroare la încărcarea modelului VAD: {e}")

def classify_audio(frame, sample_rate):
    try:
        audio_np = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        audio_tensor = torch.from_numpy(audio_np)

        if audio_tensor.shape[0] < 512:
            return "TĂCERE"

        vad_out = vad_model(audio_tensor, sample_rate).item()
        return "VORBIRE" if vad_out > 0.5 else "TĂCERE"
    except Exception as e:
        print(f"Eroare în clasificarea audio: {e}")
        return "TĂCERE"

def process_audio_file(file_path, frame_rate, update_signal, progress_signal):
    results = []
    total_frames = 0
    try:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            total_frames = wf.getnframes()

            frame_size = 512
            while True:
                frame = wf.readframes(frame_size)
                if len(frame) < frame_size * 2:
                    break

                result = classify_audio(frame, sample_rate)
                results.append(result)

                update_signal.emit(result)
                progress = int((wf.tell() / total_frames) * 100)
                progress_signal.emit(progress)

                time.sleep(0.03)

        return results, total_frames
    except Exception as e:
        print(f"Eroare în procesarea fișierului audio: {e}")
        return [], total_frames

def process_audio_file_silent(file_path):
    from types import SimpleNamespace

    dummy_signal = SimpleNamespace()
    dummy_signal.emit = lambda x: None

    with wave.open(file_path, 'rb') as wf:
        sample_rate = wf.getframerate()

    return process_audio_file(file_path, sample_rate, dummy_signal, dummy_signal)