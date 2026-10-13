import gradio as gr
import time
import threading
import numpy as np
import os
import cv2
import pygame
from ultralytics import YOLO
from scripts.llm_assistant import FolkInAssistant

# --- 1. Konfigurasi Suara ---
ALERT_AUDIO_FILE = "alert.wav"

pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound(ALERT_AUDIO_FILE)
except Exception as e:
    print(f"Error memuat audio: {e}")
    alert_sound = None

print("--- Memulai Aplikasi FolkIn ---")
print("Memuat model YOLO...")
yolo_model = YOLO("models/best.pt")

print("Memuat model Qwen...")
qwen_ai = FolkInAssistant(adapter_path="./models/folkin-qwen-lora")

# --- Variabel Global ---
last_alert_time = 0
alert_channel = None
phone_detect_start_time = None
COOLDOWN_SECONDS = 20
STABLE_SECONDS = 1.5
current_ai_message = "Sistem siap. Mulai rekam kamera untuk mengaktifkan AI."
is_generating_text = False

last_yolo_time = 0
YOLO_INTERVAL = 0.2
last_boxes = []

# --- CSS & Desain UI Peringatan (HTML) ---
CUSTOM_CSS = """
.timer, .record-time, span.time, .timestamp, [data-testid="timer"] { display: none !important; }

@keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
.alert-box {
    background-color: #c62828; color: white; padding: 40px; border-radius: 15px;
    text-align: center; border: 8px solid #ff5252; animation: blink 0.8s infinite;
    box-shadow: 0px 0px 30px rgba(255, 0, 0, 0.8);
}
"""

HTML_FOKUS = """
<div style='background-color: #2e7d32; color: white; padding: 15px; border-radius: 10px; text-align: center;'>
    <h2 style='margin: 0;'>🟢 SEDANG FOKUS</h2>
    <p style='margin: 5px 0 0 0;'>Pantauan AI Aktif.</p>
</div>
"""

HTML_WARNING = """
<div style='background-color: #f57c00; color: white; padding: 15px; border-radius: 10px; text-align: center;'>
    <h2 style='margin: 0;'>⚠️ MENDETEKSI GANGGUAN...</h2>
    <p style='margin: 5px 0 0 0;'>Letakkan HP-mu!</p>
</div>
"""

HTML_ALERT = """
<div class='alert-box'>
    <h1 style='margin: 0; font-size: 45px;'>🚨 DISTRAKSI TERDETEKSI! 🚨</h1>
    <p style='margin: 15px 0 0 0; font-size: 22px;'><b>SIMPAN HP-MU SEKARANG JUGA!</b></p>
</div>
"""


def generate_qwen_text(instruction, input_text):
    global current_ai_message, is_generating_text
    is_generating_text = True
    try:
        current_ai_message = qwen_ai.generate_response(instruction, input_text)
    except Exception as e:
        print("Error Qwen:", e)
    finally:
        is_generating_text = False


def stop_alert_sound():
    global alert_channel
    if alert_channel is not None:
        alert_channel.stop()
        alert_channel = None


def process_frame(frame, user_task):
    global last_alert_time, phone_detect_start_time, current_ai_message
    global last_yolo_time, last_boxes, alert_channel

    if frame is None:
        stop_alert_sound()
        return None, current_ai_message, HTML_FOKUS

    current_time = time.time()

    if current_time - last_yolo_time > YOLO_INTERVAL:
        results = yolo_model(frame, verbose=False, imgsz=320)
        last_boxes = results[0].boxes
        last_yolo_time = current_time

    annotated_frame = np.copy(frame)
    is_holding_phone = False

    for box in last_boxes:
        class_id = int(box.cls[0])
        if class_id < len(yolo_model.names) and yolo_model.names[class_id] == "phone_in_hand":
            is_holding_phone = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(annotated_frame, "phone_in_hand", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            break

    ui_status = HTML_FOKUS

    if is_holding_phone:
        if phone_detect_start_time is None:
            phone_detect_start_time = current_time

        elapsed_hold_time = current_time - phone_detect_start_time

        if elapsed_hold_time > 0.5:
            ui_status = HTML_WARNING

        if elapsed_hold_time > STABLE_SECONDS:
            ui_status = HTML_ALERT
            time_since_last_alert = current_time - last_alert_time

            if alert_sound and (alert_channel is None or not alert_channel.get_busy()):
                alert_channel = alert_sound.play(loops=-1)

            if time_since_last_alert > COOLDOWN_SECONDS:
                last_alert_time = current_time

                if not is_generating_text:
                    instruction = "HP terdeteksi sedang digenggam saat sesi belajar."
                    input_text = f"Tugas pengguna: {user_task}" if user_task.strip(
                    ) else "Pengguna sedang belajar."
                    threading.Thread(target=generate_qwen_text, args=(
                        instruction, input_text)).start()
    else:
        # Jika HP sudah disingkirkan, reset timer dan matikan suara alarm
        phone_detect_start_time = None
        stop_alert_sound()
        # Opsional: Jika ingin cooldown direset langsung ketika HP disingkirkan, uncomment baris bawah ini:
        # last_alert_time = 0

    return annotated_frame, current_ai_message, ui_status


# --- Antarmuka UI Gradio ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title="FolkIn - AI Focus", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🎯 FolkIn: Your Personal Focus Assistant")

    with gr.Row():
        with gr.Column(scale=2):
            camera = gr.Image(
                sources=["webcam"], streaming=True, label="Live Focus Monitor", show_label=False)

        with gr.Column(scale=1):
            status_ui = gr.HTML(value=HTML_FOKUS)

            user_task = gr.Textbox(label="Apa yang ingin kamu selesaikan sekarang?",
                                   placeholder="Contoh: Mengerjakan UI Vitaro...")
            ai_output = gr.Textbox(label="Pesan dari FolkIn AI 🤖",
                                   value=current_ai_message, interactive=False, lines=4)

    camera.stream(
        fn=process_frame,
        inputs=[camera, user_task],
        outputs=[camera, ai_output, status_ui],
        stream_every=0.03
    )

if __name__ == "__main__":
    demo.launch()
