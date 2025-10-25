import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

OUT_CSV = "sign_data.csv"

def extract_landmarks(results, image_shape):
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    h, w = image_shape
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    # normalize relative to wrist
    base_x, base_y, base_z = coords[0], coords[1], coords[2]
    norm = []
    for i in range(0, len(coords), 3):
        norm.extend([coords[i] - base_x, coords[i+1] - base_y, coords[i+2] - base_z])
    return norm

def ensure_csv_exists():
    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"f{i}" for i in range(63)] + ["label"])

def main():
    ensure_csv_exists()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Press 'q' to quit. Press 'c' to collect a sample for the current label.")
    label = input("Enter label to collect (e.g. A, B, hello): ").strip()
    sample_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collect Data - Press c to capture sample", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            label = input("Enter new label: ").strip()
            sample_count = 0
            print(f"Changed label to '{label}'.")
        elif key == ord('c'):
            print("Hold pose...")

            # Show on-screen countdown
            for i in range(3, 0, -1):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, str(i), (250, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
                cv2.putText(frame, "Capturing in...", (130, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.imshow("Collect Data - Press c to capture sample", frame)
                cv2.waitKey(1000)

            # Capture frame after countdown
            ret2, frame2 = cap.read()
            frame2 = cv2.flip(frame2, 1)
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            results2 = hands.process(rgb2)
            feat = extract_landmarks(results2, frame2.shape[:2])
            if feat is None:
                print("No hand detected. Try again.")
            else:
                feat = feat[:63] + [0.0] * max(0, 63 - len(feat))
                with open(OUT_CSV, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(feat + [label])
                sample_count += 1
                print(f"✅ Sample #{sample_count} saved for label '{label}'")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
