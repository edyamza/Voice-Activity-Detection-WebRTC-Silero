import webrtcvad
import pyaudio
import numpy as np

# Configurare WebRTC VAD
vad = webrtcvad.Vad(3)  # Nivel ridicat de detecție (mai puține false positive)

# Configurare audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30  # Cadre de 30ms (WebRTC acceptă 10, 20 sau 30ms)
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # Număr de mostre per cadru

def classify_audio(frame):
    """Clasifică cadrul audio ca Vorbire sau Tăcere"""
    is_speech = vad.is_speech(frame, RATE)

    # Convertim cadrul într-un array pentru analiză amplitudine
    audio_data = np.frombuffer(frame, dtype=np.int16)
    amplitude = np.abs(audio_data).mean()

    if is_speech :  # Vorbire clară
        return "VORBIRE"
    elif amplitude < 200: 
        return "TĂCERE" # Prag pentru tăcere
   
def vad_live_generator():
    """Generator pentru detectarea live a vocii, returnează starea pentru GUI"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=FRAME_SIZE)

    try:
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            yield classify_audio(frame)
    except Exception as e:
        print(f"Eroare: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
