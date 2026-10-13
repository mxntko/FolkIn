import argparse
import time
import cv2
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None


def main():
    parser = argparse.ArgumentParser(description="FolkIn webcam phone-in-hand detector")
    parser.add_argument("--model", default="best.pt", help="Path to trained best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--stable-seconds", type=float, default=1.5,
                        help="How long phone_in_hand must persist before alert")
    parser.add_argument("--cooldown", type=float, default=15.0,
                        help="Minimum seconds between alerts")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Webcam tidak dapat dibuka. Tutup aplikasi lain yang memakai kamera.")

    class_names = model.names
    phone_in_hand_id = next(
        (int(k) for k, v in class_names.items() if str(v).lower() == "phone_in_hand"),
        0
    )
    first_detected_at = None
    last_alert_at = 0.0

    print("FolkIn webcam aktif. Tekan Q pada jendela kamera untuk berhenti.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame webcam gagal dibaca.")
                break

            frame = cv2.flip(frame, 1)
            result = model.predict(frame, conf=args.confidence, imgsz=512, verbose=False)[0]
            annotated = result.plot()

            found = False
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls.item()) == phone_in_hand_id:
                        found = True
                        break

            now = time.time()
            if found:
                if first_detected_at is None:
                    first_detected_at = now
                stable = now - first_detected_at
                status = f"PHONE IN HAND: {stable:.1f}s"
                if stable >= args.stable_seconds and now - last_alert_at >= args.cooldown:
                    print("PERINGATAN: HP terdeteksi sedang digenggam. Kembali ke tugas!")
                    if winsound is not None:
                        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                    last_alert_at = now
            else:
                first_detected_at = None
                status = "FOCUSED / NO PHONE-IN-HAND DETECTED"

            color = (0, 0, 255) if found else (0, 180, 0)
            cv2.rectangle(annotated, (8, 8), (annotated.shape[1]-8, 52), color, 2)
            cv2.putText(annotated, status, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)
            cv2.imshow("FolkIn - Focus Monitor (Q to quit)", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
