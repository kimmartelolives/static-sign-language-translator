import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque
from gtts import gTTS
import threading
import os
import pygame

# ---------------------------
# Load Model
# ---------------------------
MODEL_FILE = "sign_model.joblib"
clf = joblib.load(MODEL_FILE)

# ---------------------------
# Mediapipe Setup
# ---------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# ---------------------------
# Helper Functions
# ---------------------------
def extract_landmarks(results, image_shape):
    """Extract normalized 3D hand landmark coordinates."""
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    h, w = image_shape
    coords = []
    for lm in hand.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    # normalize based on wrist (landmark 0)
    base_x, base_y, base_z = coords[:3]
    norm = []
    for i in range(0, len(coords), 3):
        norm.extend([
            coords[i] - base_x,
            coords[i + 1] - base_y,
            coords[i + 2] - base_z
        ])
    return norm


def speak_text(text):
    """Convert text to speech using gTTS and play safely from local folder."""
    def speak_thread(txt):
        try:
            # Ensure tts_audio folder exists
            audio_dir = "tts_audio"
            os.makedirs(audio_dir, exist_ok=True)

            txt = str(txt)
            filename = os.path.join(audio_dir, f"{txt}.mp3")

            # Always re-generate (in case old file is corrupted)
            if os.path.exists(filename):
                os.remove(filename)

            # Generate MP3
            tts = gTTS(text=txt, lang='en')
            tts.save(filename)

            # Small delay to ensure file is fully written
            time.sleep(0.3)

            # Validate file size (avoid corrupt 0-byte files)
            if os.path.getsize(filename) < 1000:
                print(f"TTS Error: {filename} seems incomplete, regenerating...")
                os.remove(filename)
                tts = gTTS(text=txt, lang='en')
                tts.save(filename)

            # Initialize and play audio
            pygame.mixer.quit()  # reset in case of leftover state
            pygame.mixer.init()
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            print(f"TTS Error: {e}")

    threading.Thread(target=speak_thread, args=(text,), daemon=True).start()


# ---------------------------
# Main Loop
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("🖐 Starting Sign Language Translator")
    print("Press 'q' to quit")

    history = deque(maxlen=5)  # smoothing buffer
    last_spoken = None
    last_spoken_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS)

            feat = extract_landmarks(results, frame.shape[:2])
            if feat:
                feat = feat[:63] + [0.0] * max(0, 63 - len(feat))
                pred = clf.predict([feat])[0]

                # Convert to string (avoid numpy.int64)
                pred_str = str(pred)

                history.append(pred_str)
                stable_pred = max(set(history), key=history.count)

                # Display prediction
                cv2.putText(frame, f"Prediction: {stable_pred}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Speak only when new or cooldown passed
                now = time.time()
                if stable_pred != last_spoken or now - last_spoken_time > 3:
                    print(f"Predicted: {stable_pred}")
                    speak_text(stable_pred)
                    last_spoken = stable_pred
                    last_spoken_time = now
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Sign Language Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
