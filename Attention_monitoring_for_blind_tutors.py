import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import time
import os
import threading
from ultralytics import YOLO

def speak(text):
    command = f'powershell -Command "Add-Type -AssemblyName System.Speech; ' \
              f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
    os.system(command)

def speak_async(text):
    threading.Thread(target=speak, args=(text,), daemon=True).start()

YOLO_MODEL_PATH = "yolov8n.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

PHONE_CLASS_ID = 67
CUSTOM_PHONE_CLASS_NAME = "phone"  

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

UPPER_LIP = 13
LOWER_LIP = 14
LEFT_LIP  = 78
RIGHT_LIP = 308

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_phone(frame, model, custom_model=True):
    results = model(frame, verbose=False)[0]
    phone_detected = False
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        cls_name = model.names[cls_id].lower()
        is_phone = False
        if custom_model:
            if CUSTOM_PHONE_CLASS_NAME.lower() in cls_name:
                is_phone = True
        else:
            if cls_id == PHONE_CLASS_ID:
                is_phone = True
        if is_phone and conf > 0.45:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Phone {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
    return phone_detected, frame

cap = cv2.VideoCapture(0)

EAR_THRESHOLD    = 0.25
DROWSY_FRAMES    = 20
DISTRACTION_TIME = 2
ANGLE_THRESHOLD  = 40
ABSENT_TIME      = 3
MOUTH_OPEN_RATIO = 0.15
SPEAKING_FRAMES  = 12
ALERT_INTERVAL   = 5

PHONE_DETECT_EVERY_N_FRAMES = 5

frame_count           = 0
global_frame_counter  = 0
no_face_timer         = time.time()
distraction_start_time = None
last_alert_time       = 0
absent_flag           = False
speaking_counter      = 0
phone_detected_flag   = False

using_custom_model = os.path.exists(YOLO_MODEL_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    global_frame_counter += 1
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results     = face_mesh.process(rgb)
    status_list = []
    face_detected = False
    if global_frame_counter % PHONE_DETECT_EVERY_N_FRAMES == 0:
        phone_detected_flag, frame = detect_phone(
            frame, yolo_model, custom_model=using_custom_model
        )
    if phone_detected_flag:
        status_list.append("Phone")
    if results.multi_face_landmarks:
        face_detected = True
        no_face_timer = time.time()
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_eye = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in LEFT_EYE
            ]
            right_eye = [
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in RIGHT_EYE
            ]
            ear = (
                eye_aspect_ratio(left_eye) +
                eye_aspect_ratio(right_eye)
            ) / 2.0
                
            nose       = landmarks[1]
            left_face  = landmarks[234]
            right_face = landmarks[454]

            nose_x  = int(nose.x * w)
            left_x  = int(left_face.x * w)
            right_x = int(right_face.x * w)
            center  = (left_x + right_x) // 2
            deviation = abs(nose_x - center)

            if deviation < 20:
                direction = "Forward"
            elif nose_x < center:
                direction = "Left"
            else:
                direction = "Right"

            if ear < EAR_THRESHOLD:
                frame_count += 1
            else:
                frame_count = 0
            current_time     = time.time()
            is_distracted_now = (deviation > ANGLE_THRESHOLD or direction != "Forward")

            if is_distracted_now:
                if distraction_start_time is None:
                    distraction_start_time = current_time
            else:
                distraction_start_time = None

            is_distracted_final = (
                distraction_start_time is not None and
                current_time - distraction_start_time > DISTRACTION_TIME
            )
            upper_lip   = landmarks[UPPER_LIP]
            lower_lip   = landmarks[LOWER_LIP]
            left_lip    = landmarks[LEFT_LIP]
            right_lip_l = landmarks[RIGHT_LIP]

            upper_y     = int(upper_lip.y * h)
            lower_y     = int(lower_lip.y * h)
            left_lip_x  = int(left_lip.x * w)
            right_lip_x = int(right_lip_l.x * w)

            mouth_open  = abs(lower_y - upper_y)
            mouth_width = abs(right_lip_x - left_lip_x)
            mouth_ratio = mouth_open / mouth_width if mouth_width > 0 else 0

            if mouth_ratio > MOUTH_OPEN_RATIO:
                speaking_counter += 1
            else:
                speaking_counter = max(0, speaking_counter - 1)
            is_speaking = speaking_counter > SPEAKING_FRAMES

            if phone_detected_flag:
                status = "Using Phone"
                color  = (0, 0, 255)
            elif frame_count > DROWSY_FRAMES:
                status = "Drowsy"
                color  = (0, 0, 255)
            elif is_distracted_final:
                status = "Distracted"
                color  = (0, 255, 255)
            elif is_speaking:
                status = "Speaking"
                color  = (255, 0, 0)
            else:
                status = "Attentive"
                color  = (0, 255, 0)
            if status not in status_list:
                status_list.append(status)

            cv2.putText(
                frame, f"Status : {status}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

    if not face_detected:
        if time.time() - no_face_timer > ABSENT_TIME:
            absent_flag  = True
            status_list  = ["Absent"]
            cv2.putText(
                frame, "Student Absent",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
    else:
        absent_flag = False

    current_time = time.time()

    if current_time - last_alert_time >= ALERT_INTERVAL:

        if absent_flag:
            message = "Student not in frame"

        elif "Using Phone" in status_list or "Phone" in status_list:
            message = "Student is using a mobile phone"

        elif "Drowsy" in status_list:
            message = "Student is drowsy"

        elif "Distracted" in status_list:
            message = "Student is distracted"

        elif "Speaking" in status_list:
            message = "Student is speaking"

        elif len(status_list) > 0:
            message = "Student is attentive"

        else:
            message = "No student detected"

        speak_async(message)
        last_alert_time = current_time

    cv2.imshow("Attention Monitoring System for Blind Tutors", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()