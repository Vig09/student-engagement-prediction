import cv2
import dlib
from scipy.spatial import distance
import time
import numpy as np

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to detect head direction
def get_head_direction(landmarks):
    nose_point = (landmarks.part(30).x, landmarks.part(30).y)
    chin_point = (landmarks.part(8).x, landmarks.part(8).y)
    left_cheek = (landmarks.part(1).x, landmarks.part(1).y)
    right_cheek = (landmarks.part(15).x, landmarks.part(15).y)

    face_width = distance.euclidean(left_cheek, right_cheek)
    nose_to_left = distance.euclidean(nose_point, left_cheek)
    nose_to_right = distance.euclidean(nose_point, right_cheek)

    if nose_to_left / face_width < 0.35:
        return "Right"
    elif nose_to_right / face_width < 0.35:
        return "Left"
    else:
        return "Center"

# Load models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EYE_AR_THRESH = 0.25
BLINK_CHECK_INTERVAL = 3
ENGAGEMENT_CHECK_INTERVAL = 10
HEAD_TURN_THRESHOLD = 2  # seconds

# Initialize
start_time = time.time()
last_blink_check = time.time()
head_turn_start = None

# Status tracking
face_detected_in_10s = False
blink_count_in_10s = 0
engagement_status_text = "Checking..."
blink_status_text = "Waiting..."
engagement_reason = ""
head_direction = "Unknown"
turned_too_long = False

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_detected = False
    blink_now = False

    if len(faces) > 0:
        face_detected = True
        face_detected_in_10s = True

        for face in faces:
            landmarks = predictor(gray, face)

            left_eye = [landmarks.part(i) for i in range(36, 42)]
            right_eye = [landmarks.part(i) for i in range(42, 48)]
            left_pts = [(p.x, p.y) for p in left_eye]
            right_pts = [(p.x, p.y) for p in right_eye]

            left_ear = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < EYE_AR_THRESH:
                blink_now = True
                blink_count_in_10s += 1

            head_direction = get_head_direction(landmarks)
            if head_direction != "Center":
                if head_turn_start is None:
                    head_turn_start = time.time()
                elif time.time() - head_turn_start > HEAD_TURN_THRESHOLD:
                    turned_too_long = True
            else:
                head_turn_start = None
                turned_too_long = False

            for (x, y) in left_pts + right_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Blink check every 3s
    current_time = time.time()
    if current_time - last_blink_check >= BLINK_CHECK_INTERVAL:
        blink_status_text = "Blinking" if blink_now else "Not Blinking"
        print(f"[{int(current_time)}s] Blink Status: {blink_status_text}")
        last_blink_check = current_time

    # Engagement check every 10s
    if current_time - start_time >= ENGAGEMENT_CHECK_INTERVAL:
        print("\n--- Engagement Check ---")
        print("Face Detected:", face_detected_in_10s)
        print("Blinks in 10s:", blink_count_in_10s)
        print("Head Direction:", head_direction)
        print("Head turned too long:", turned_too_long)

        if not face_detected_in_10s:
            engagement_status_text = "❌ Not Engaged"
            engagement_reason = "Reason: No Face Detected"
        elif blink_count_in_10s < 2:
            engagement_status_text = "❌ Not Engaged"
            engagement_reason = "Reason: Blinks < 2"
        elif turned_too_long:
            engagement_status_text = "❌ Not Engaged"
            engagement_reason = f"Reason: Head turned {head_direction} > 2s"
        else:
            engagement_status_text = "✅ Engaged"
            engagement_reason = "Reason: Face Detected, Blinking, Head Centered"

        print("Engagement:", engagement_status_text)
        print(engagement_reason)
        print("--------------------------\n")

        # Reset
        blink_count_in_10s = 0
        face_detected_in_10s = False
        start_time = current_time

    # Show info on screen (updated display)
    cv2.putText(frame, f"Face: {'Detected' if face_detected else 'Not Detected'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Blink: {blink_status_text}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(frame, f"Head: {head_direction}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
    cv2.putText(frame, f"Engagement: {engagement_status_text}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if "✅" in engagement_status_text else (0, 0, 255), 2)
    cv2.putText(frame, engagement_reason, (10, 130),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)

    cv2.imshow("Student Engagement Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
