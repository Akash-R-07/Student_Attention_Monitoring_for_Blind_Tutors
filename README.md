# Attention Monitoring System for Blind Tutors

This project is a real-time computer vision system designed to help blind or visually impaired tutors monitor student attention during online or in-person sessions. It uses a webcam to detect various attention states and provides audio feedback through text-to-speech alerts.

## Available Features

- **Face & Landmark Detection** – Uses MediaPipe Face Mesh to track facial landmarks.
- **Eye Aspect Ratio (EAR)** – Detects drowsiness when eyes are closed for a prolonged period.
- **Head Pose Estimation** – Detects distraction based on head turning away from the screen.
- **Mouth Opening Ratio** – Identifies when the student is speaking.
- **Phone Usage Detection** – Uses YOLOv8 object detection to spot mobile phones.
- **Student Absence Detection** – Alerts if no face is detected for a configured time.
- **Voice Alerts** – Announces the student’s state (Attentive, Drowsy, Distracted, Using Phone, Speaking, Absent) at regular intervals.
- **Visual Feedback** – Displays status on the video feed with bounding boxes and color-coded text.
