import sys
import os
import glob
import vad_live
import vad_rec
import vad_rec_silero
from evaluation import evaluate_model, save_metrics_chart
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QFileDialog, QComboBox, QProgressBar, QTextEdit, QGraphicsDropShadowEffect)
from PyQt6.QtGui import (QPixmap, QPainter, QPen, QColor, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import simpleaudio as sa
import time
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_latest_metrics_chart(path="output"):
    files = glob.glob(os.path.join(path, "*_barchart_*.png"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

class MetricsChartWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reprezentare grafică a metricilor")
        self.setGeometry(300, 300, 600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        image_path = get_latest_metrics_chart()
        if image_path and os.path.exists(image_path):
            label = QLabel(self)
            pixmap = QPixmap(image_path)
            label.setPixmap(pixmap.scaledToWidth(580))
            layout.addWidget(label)
        else:
            layout.addWidget(QLabel("Imaginea nu a fost găsită."))

class ComparisonWindow(QWidget):
    def __init__(self, metrics_webrtc, metrics_silero, file_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comparație între modele VAD")
        self.setGeometry(350, 350, 700, 500)

        layout = QVBoxLayout()
        self.setLayout(layout)

        keys = list(metrics_webrtc.keys())
        values1 = [metrics_webrtc[k] for k in keys]
        values2 = [metrics_silero[k] for k in keys]

        x = np.arange(len(keys))
        width = 0.35

        self.fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width/2, values1, width, label='WebRTC')
        bars2 = ax.bar(x + width/2, values2, width, label='Silero')

        ax.set_ylabel('Valori')
        ax.set_title('Compararea modelelor VAD')
        ax.set_xticks(x)
        ax.set_xticklabels(keys)
        ax.legend()
        plt.tight_layout()

        # Etichete deasupra fiecărei bare WebRTC
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.2f}", ha='center', va='bottom')

        # Etichete deasupra fiecărei bare Silero
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.2f}", ha='center', va='bottom')

        canvas = FigureCanvas(self.fig)
        layout.addWidget(canvas)

        base_name = "unknown"
        if file_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs("output", exist_ok=True)
        index = 1
        while True:
            image_path = os.path.join("output", f"comp_barchart_{base_name}_{index}.png")
            if not os.path.exists(image_path):
                break
            index += 1
        self.fig.savefig(image_path)

class AudioProcessorThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, file_path, vad_method):
        super().__init__()
        self.file_path = file_path
        self.vad_method = vad_method
        self.play_obj = None
        self.frames_played = 0
        self.total_frames = 0
        self.audio_started = False
        self.stop_processing = False

    def run(self):
        with wave.open(self.file_path, 'rb') as wf:
            frame_rate = wf.getframerate()
            self.total_frames = wf.getnframes()

        if not self.audio_started:
            audio_data = sa.WaveObject.from_wave_file(self.file_path)
            self.play_obj = audio_data.play()
            self.audio_started = True

        if self.vad_method == "WebRTC":
            results = vad_rec.process_audio_file_gui(
                self.file_path,
                self.update_signal,
                self.progress_signal,
                self.stop_processing
            )
        else:
            results, played_frames = vad_rec_silero.process_audio_file(
                self.file_path,
                frame_rate,
                self.update_signal,
                self.progress_signal
            )

        self.play_obj.wait_done()
        self.frames_played = self.total_frames
        self.finished_signal.emit(self.file_path)

    def stop(self):
        self.stop_processing = True
        if self.play_obj:
            self.play_obj.stop()
        self.finished_signal.emit(self.file_path)

class MetricsWindow(QWidget):
    def __init__(self, metrics=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metrici Evaluare VAD")
        self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()
        self.metrics_box = QTextEdit()
        self.metrics_box.setReadOnly(True)
        self.metrics_box.setStyleSheet("font-family: Arial Black; font-size: 12px;")
        layout.addWidget(self.metrics_box)

        self.setLayout(layout)
        self.load_metrics(metrics)

    def load_metrics(self, metrics):
        if metrics:
            metrics_text = f"Accuracy: {metrics['accuracy']}\n"
            metrics_text += f"Precision: {metrics['precision']}\n"
            metrics_text += f"Recall: {metrics['recall']}\n"
            metrics_text += f"F1 Score: {metrics['f1_score']}\n"
            metrics_text += f"False Positive Rate: {metrics['false_positive_rate']}\n"
            metrics_text += f"False Negative Rate: {metrics['false_negative_rate']}\n"
            self.metrics_box.setPlainText(metrics_text)
        else:
            self.metrics_box.setPlainText("Nu s-au găsit metrici.")

class ConfusionMatrixWindow(QWidget):
    def __init__(self, cm_webrtc=None, cm_silero=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confusion Matrix")
        self.setGeometry(400, 400, 600, 800)

        layout = QVBoxLayout()
        self.setLayout(layout)

        fig, axs = plt.subplots(2, 1, figsize=(6, 10))

        if cm_webrtc is not None:
            axs[0].matshow(cm_webrtc, cmap='Blues')
            for (i, j), val in np.ndenumerate(cm_webrtc):
                axs[0].text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=12)
            axs[0].set_title('Confusion Matrix - WebRTC')
            axs[0].set_xlabel('Predicted')
            axs[0].set_ylabel('Actual')
            axs[0].set_xticklabels([''] + ['Silence', 'Speech'])
            axs[0].set_yticklabels([''] + ['Silence', 'Speech'])

        if cm_silero is not None:
            axs[1].matshow(cm_silero, cmap='Blues')
            for (i, j), val in np.ndenumerate(cm_silero):
                axs[1].text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=12)
            axs[1].set_title('Confusion Matrix - Silero')
            axs[1].set_xlabel('Predicted')
            axs[1].set_ylabel('Actual')
            axs[1].set_xticklabels([''] + ['Silence', 'Speech'])
            axs[1].set_yticklabels([''] + ['Silence', 'Speech'])

        fig.tight_layout(pad=4)

        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)

class VoiceActivityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Activity Detection")
        self.setGeometry(100, 100, 350, 450)

        self.vad_method = "WebRTC"
        self.initUI()
        self.vad_generator = None
        self.is_vad_running = False
        self.audio_thread = None
        self.metrics_window = None
        self.metrics_chart_window = None
        self.comparison_window = None
        self.confusion_matrix_window = None

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)
        self.content_layout = QVBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        self.vad_selector = QComboBox(self)
        self.vad_selector.addItems(["🌐 WebRTC - VAD", "🔕 Silero - VAD", "🔀 Compară ambele"])
        self.vad_selector.setCurrentIndex(2)
        self.vad_selector.currentTextChanged.connect(self.update_vad_method)
        self.content_layout.addWidget(self.vad_selector)

        self.profile_label = QLabel(self)
        self.update_profile_image("white")
        self.apply_glow_effect("white")
        self.profile_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(self.profile_label)

        self.button_row_layout = QHBoxLayout()

        self.vad_live_button = QPushButton("🎙️ Start VAD Live", self)
        self.vad_live_button.clicked.connect(self.toggle_vad_live)
        self.button_row_layout.addWidget(self.vad_live_button)

        self.vad_file_button = QPushButton("📂 Selectează fișier audio", self)
        self.vad_file_button.clicked.connect(self.process_audio_file)
        self.button_row_layout.addWidget(self.vad_file_button)

        self.audio_playing_button = QPushButton("🔊 Redare Audio...", self)
        self.audio_playing_button.setEnabled(False)
        self.audio_playing_button.hide()
        self.button_row_layout.addWidget(self.audio_playing_button)

        self.view_results_button = QPushButton("📊 Vezi rezultatele", self)
        self.view_results_button.clicked.connect(self.show_results)
        self.view_results_button.hide()
        self.button_row_layout.addWidget(self.view_results_button)

        self.content_layout.addLayout(self.button_row_layout)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.main_layout.addWidget(self.progress_bar)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)

        self.update_vad_method(self.vad_selector.currentText())

    def update_vad_method(self, method):
        if "WebRTC" in method:
            self.vad_method = "WebRTC"
            self.vad_live_button.show()
            self.vad_file_button.show()
        elif "Silero" in method:
            self.vad_method = "Silero"
            self.vad_live_button.hide()
            self.vad_file_button.show()
        elif "Compară" in method:
            self.vad_method = "COMPARE"
            self.vad_live_button.hide()
            self.vad_file_button.show()

    def toggle_vad_live(self):
        if self.is_vad_running:
            self.stop_vad_live()
        else:
            self.start_vad_live()

    def start_vad_live(self):
        self.vad_generator = vad_live.vad_live_generator()
        self.timer.start(200)
        self.vad_live_button.setText("🛑 Stop VAD Live")
        self.is_vad_running = True

    def stop_vad_live(self):
        self.timer.stop()
        self.vad_live_button.setText("🎙️ Start VAD Live")
        self.is_vad_running = False

    def update_status(self):
        if self.vad_generator:
            try:
                status = next(self.vad_generator)
                color_map = {"VORBIRE": "green", "TĂCERE": "white"}
                status_color = color_map.get(status, "white")
                self.update_profile_image(status_color)
                self.apply_glow_effect(status_color)
            except StopIteration:
                self.timer.stop()

    def process_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selectează un fișier audio", "", "Audio Files (*.wav)")
        if file_path:
            self.audio_thread = AudioProcessorThread(file_path, self.vad_method)
            self.audio_thread.update_signal.connect(self.show_audio_results)
            self.audio_thread.progress_signal.connect(self.update_progress_bar)
            self.audio_thread.finished_signal.connect(self.enable_view_results_button)
            self.audio_thread.start()

            self.vad_file_button.hide()
            self.audio_playing_button.show()

    def enable_view_results_button(self):
        self.audio_playing_button.hide()
        self.view_results_button.show()

    def show_results(self):
        file_path = self.audio_thread.file_path

        if self.vad_method == "COMPARE":
            self.show_comparison_spectrograms(file_path)

            metrics_webrtc, cm_webrtc = evaluate_model(file_path, model="webrtc")
            metrics_silero, cm_silero = evaluate_model(file_path, model="silero")

            self.comparison_window = ComparisonWindow(metrics_webrtc, metrics_silero, file_path)
            self.comparison_window.show()

            self.confusion_matrix_window = ConfusionMatrixWindow(cm_webrtc=cm_webrtc, cm_silero=cm_silero)
            self.confusion_matrix_window.show()

        else:
            self.show_spectrogram(file_path)

            metrics, cm = evaluate_model(file_path, model=self.vad_method.lower())
            timestamp = time.strftime("%d_%H%M")
            save_metrics_chart(metrics, model=self.vad_method.lower())

            self.metrics_window = MetricsWindow(metrics=metrics)
            self.metrics_window.show()
            self.metrics_chart_window = MetricsChartWindow()
            self.metrics_chart_window.show()

            self.confusion_matrix_window = ConfusionMatrixWindow(cm_webrtc=cm)
            self.confusion_matrix_window.show()

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def show_audio_results(self, status):
        color_map = {"VORBIRE": "green", "TĂCERE": "white"}
        status_color = color_map.get(status, "white")
        self.update_profile_image(status_color)
        self.apply_glow_effect(status_color)

    def update_profile_image(self, color):
        img_path = "/Users/daemondog/Desktop/proiect PSV - VAD/images.png"
        if not os.path.exists(img_path):
            print(f"[Eroare] Imaginea nu a fost găsită: {img_path}")
            return

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            print("[Eroare] QPixmap este null.")
            return

        size = 150
        pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)

        circular_pixmap = QPixmap(size, size)
        circular_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(circular_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addEllipse(0, 0, size, size)
        painter.setClipPath(path)

        painter.drawPixmap(0, 0, pixmap)
        painter.setPen(QPen(QColor(color), 6))
        painter.drawEllipse(3, 3, size - 6, size - 6)
        painter.end()

        self.profile_label.setPixmap(circular_pixmap)

    def reset_ui(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        if self.metrics_window:
            self.metrics_window.close()
        if self.metrics_chart_window:
            self.metrics_chart_window.close()
        if self.comparison_window:
            self.comparison_window.close()
        if self.confusion_matrix_window:
            self.confusion_matrix_window.close()
        self.initUI()
        self.resize(350, 450)

    def apply_glow_effect(self, color):
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(30)
        glow.setOffset(0)
        glow.setColor(QColor(color))
        self.profile_label.setGraphicsEffect(glow)

    def show_spectrogram(self, file_path):
        self.profile_label.hide()
        self.vad_live_button.hide()
        self.vad_file_button.hide()
        self.vad_selector.hide()
        self.view_results_button.hide()

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(self.vad_method.upper(), fontsize=14)

        with wave.open(file_path, 'rb') as wf:
            rate = wf.getframerate()
            total_frames = wf.getnframes()
            wf.rewind()

            if not self.audio_thread or self.audio_thread.frames_played == 0:
                print("[Eroare] Niciun cadru redat.")
                return

            frames_to_read = min(self.audio_thread.frames_played, total_frames)
            signal = np.frombuffer(wf.readframes(frames_to_read), dtype=np.int16)

            if signal.size == 0:
                print("[Eroare] Semnalul audio este gol.")
                return

        ax[0].specgram(signal, Fs=rate, cmap='plasma')
        ax[0].set_title('Spectrograma')

        ax[1].plot(np.linspace(0, len(signal) / rate, num=len(signal)), signal)
        ax[1].set_title('Waveform')

        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.content_layout.addStretch(1)

        self.back_button = QPushButton("⬅️ Înapoi", self)
        self.back_button.clicked.connect(self.reset_ui)
        self.main_layout.addWidget(self.back_button)
        self.resize(1000, 800)

    def show_comparison_spectrograms(self, file_path):
        self.profile_label.hide()
        self.vad_live_button.hide()
        self.vad_file_button.hide()
        self.vad_selector.hide()
        self.view_results_button.hide()

        fig, axs = plt.subplots(4, 1, figsize=(12, 12))
        fig.suptitle("COMPARAȚIE WEBRTC VS SILERO", fontsize=14)

        for idx, model_name in enumerate(["WebRTC", "Silero"]):
            with wave.open(file_path, 'rb') as wf:
                rate = wf.getframerate()
                total_frames = wf.getnframes()
                wf.rewind()

                signal = np.frombuffer(wf.readframes(total_frames), dtype=np.int16)

            axs[2 * idx].specgram(signal, Fs=rate, cmap='plasma')
            axs[2 * idx + 1].plot(np.linspace(0, len(signal) / rate, num=len(signal)), signal)

        plt.tight_layout()
        canvas = FigureCanvas(fig)
        self.content_layout.addWidget(canvas)
        self.content_layout.addStretch(1)

        self.back_button = QPushButton("⬅️ Înapoi", self)
        self.back_button.clicked.connect(self.reset_ui)
        self.main_layout.addWidget(self.back_button)
        self.resize(1000, 1000)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceActivityGUI()
    window.show()
    sys.exit(app.exec())
