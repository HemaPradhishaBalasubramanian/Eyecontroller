import cv2
import mediapipe as mp
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

blink_start = None  # Track when blink started

def get_ear(landmarks, eye_indices):
    left = landmarks[eye_indices[0]]
    right = landmarks[eye_indices[3]]
    top = (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2
    bottom = (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2
    ear = abs(top - bottom) / abs(left.x - right.x)
    return ear

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Draw green dots on eye corners
            for idx in [33, 133, 362, 263]:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                left_eye = [33, 160, 158, 133, 153, 144]
            right_eye = [362, 385, 387, 263, 373, 380]

            left_ear = get_ear(landmarks, left_eye)
            right_ear = get_ear(landmarks, right_eye)
            ear = (left_ear + right_ear) / 2

            if ear < 0.2:
                if blink_start is None:
                    blink_start = time.time()
            else:
                if blink_start is not None:
                    blink_duration = time.time() - blink_start
                    if blink_duration < 1.0:
                        pyautogui.press('space')  # Short blink: play/pause
                    elif blink_duration >= 1.5:
                        pyautogui.press('right')  # Long blink: skip forward
                    blink_start = None

    cv2.imshow("Eye Blink Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
            