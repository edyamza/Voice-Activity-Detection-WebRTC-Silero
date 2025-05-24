import webrtcvad
import numpy as np
import wave
import json
import os

# Inițializare VAD cu sensibilitate maximă
vad = webrtcvad.Vad(3)

# Praguri
AMPLITUDE_THRESHOLD = 200
DURATION_THRESHOLD = 0.5  # secunde

def classify_audio(frame, rate, history, prev_amplitude):
    audio_data = np.frombuffer(frame, dtype=np.int16)
    amplitude = np.abs(audio_data).mean()

    history.append(amplitude)
    max_history_length = int(rate * DURATION_THRESHOLD / 1000 / 0.03)

    if len(history) > max_history_length:
        history.pop(0)

    if np.abs(amplitude - prev_amplitude) > 50 and amplitude > AMPLITUDE_THRESHOLD:
        return "VORBIRE"
    return "TĂCERE"

def process_audio_file(file_path):
    results = []
    history = []
    prev_amplitude = 0

    with wave.open(file_path, 'rb') as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        frame_duration_ms = 30
        frame_size = int(rate * frame_duration_ms / 1000)

        if width != 2 or channels != 1:
            raise ValueError("Fișierul audio trebuie să fie mono, 16-bit PCM.")

        while True:
            frame = wf.readframes(frame_size)
            if len(frame) < frame_size * width:
                break

            is_speech = vad.is_speech(frame, rate)
            if is_speech:
                results.append("VORBIRE")
            else:
                result = classify_audio(frame, rate, history, prev_amplitude)
                results.append(result)
            prev_amplitude = np.abs(np.frombuffer(frame, dtype=np.int16)).mean()

    return results

if __name__ == "__main__":
    input_folder = "/Users/daemondog/Desktop/proiect PSV - VAD/AUDIO FILES"
    output_json = "/Users/daemondog/Desktop/proiect PSV - VAD/generareetichete/etic.json"

    # Încercăm să citim fișierul JSON existent
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    for i in range(31, 101):
        audio_file = os.path.join(input_folder, f"audio{i}.wav")
        if not os.path.exists(audio_file):
            print(f"⚠️ Fișier inexistent: {audio_file}")
            continue

        etichete = process_audio_file(audio_file)
        data[audio_file] = etichete
        print(f"✅ Etichete generate pentru: {audio_file}")

    # Scriem înapoi fișierul JSON completat
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"🎉 Etichetele au fost salvate în {output_json}")
