import os
import wave
import json
import vad_rec
import vad_rec_silero
import matplotlib.pyplot as plt
import datetime
import numpy as np

def citire_etichete(fisier_audio, fisier_json="/Users/daemondog/Desktop/proiect PSV - VAD/etichete_audio.json"):
    with open(fisier_json, "r", encoding="utf-8") as f:
        etichete = json.load(f)
    return etichete.get(fisier_audio, None)

def evaluate_model(audio_path, model="webrtc", frames_played=None, etichete_json="/Users/daemondog/Desktop/proiect PSV - VAD/etichete_audio.json"):
    if model == "webrtc":
        results = vad_rec.process_audio_file_silent(audio_path)
        frames_read = None
    else:
        results, frames_read = vad_rec_silero.process_audio_file_silent(audio_path)

    with wave.open(audio_path, 'rb') as wf:
        total_frames = wf.getnframes()
        frame_rate = wf.getframerate()
        if frames_played:
            played_duration = frames_played / frame_rate
        elif frames_read:
            played_duration = frames_read / frame_rate
        else:
            played_duration = total_frames / frame_rate

    estimated_labels = results
    ref_labels = citire_etichete(os.path.basename(audio_path), etichete_json)
    if ref_labels is None:
        raise ValueError(f"Etichetele pentru {os.path.basename(audio_path)} nu au fost găsite.")

    min_len = min(len(ref_labels), len(estimated_labels))
    estimated_labels = estimated_labels[:min_len]
    ref_labels = ref_labels[:min_len]

    TP = sum(p == "VORBIRE" and t == "VORBIRE" for p, t in zip(estimated_labels, ref_labels))
    TN = sum(p == "TĂCERE" and t == "TĂCERE" for p, t in zip(estimated_labels, ref_labels))
    FP = sum(p == "VORBIRE" and t == "TĂCERE" for p, t in zip(estimated_labels, ref_labels))
    FN = sum(p == "TĂCERE" and t == "VORBIRE" for p, t in zip(estimated_labels, ref_labels))

    total = TP + TN + FP + FN
    if total == 0:
        accuracy = precision = recall = f1_score = fpr = fnr = 0
    else:
        accuracy = (TP + TN) / total
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
        fnr = FN / (TP + FN) if (TP + FN) > 0 else 0

    # Matricea de Confuzie
    confusion_mat = np.array([[TN, FP],
                              [FN, TP]])

    # Scriere rezultate în output.txt
    results_text = f"Fișier: {os.path.basename(audio_path)}\n"
    results_text += f"Model: {model.upper()}\n"
    results_text += "-------- Confusion Matrix --------\n"
    results_text += f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}\n"
    results_text += "-------- Metrici --------\n"
    results_text += f"Accuracy               = {accuracy:.4f}\n"
    results_text += f"Precision              = {precision:.4f}\n"
    results_text += f"Recall                 = {recall:.4f}\n"
    results_text += f"F1-score               = {f1_score:.4f}\n"
    results_text += f"False Positive Rate    = {fpr:.4f}\n"
    results_text += f"False Negative Rate    = {fnr:.4f}\n"
    results_text += "----------------------------------\n\n"

    output_path = os.path.join(os.path.dirname(__file__), "output.txt")
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as out:
            existing_content = out.read()
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(results_text + existing_content)
    else:
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(results_text)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr
    }, confusion_mat

def save_metrics_chart(metrics: dict, model: str = "webrtc", output_path: str = None):
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%d_%H%M")
        prefix = "webrtc" if model == "webrtc" else "silero" if model == "silero" else "comp"
        output_path = f"output/{prefix}_barchart_{timestamp}.png"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylim(0, 1.0)
    plt.title("Evaluare VAD")
    plt.ylabel("Valoare")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f"{height:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()