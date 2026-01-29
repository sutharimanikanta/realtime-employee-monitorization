# 🚗 Vertic AI
## Real-Time Vision-Based Drowsiness, Distraction & Microsleep Detection System

Vertic AI is a real-time computer vision system designed to monitor driver alertness and prevent fatigue-related accidents. The system detects **drowsiness, microsleep, yawning, excessive blinking, and visual distraction** using facial landmarks, head-pose estimation, and temporal reasoning.

This project emphasizes **physiologically accurate signals**, **temporal intelligence**, and **production-aware system design**.

---

## 🔍 Key Features & Technical Highlights

### 1️⃣ Eye Aspect Ratio (EAR) — Blink & Microsleep Detection
- Computes EAR using facial landmark geometry
- Short-term EAR drop → Blink  
- Sustained EAR drop → Microsleep
- Combines **time-based thresholds with frame analysis**
- Avoids naive frame-count heuristics

---

### 2️⃣ Mouth Aspect Ratio (MAR) — Yawn Detection
- Detects yawning using:
  - Vertical lip distance
  - Horizontal mouth width
  - Sustained MAR threshold
- Uses temporal validation to reduce false positives

---

### 3️⃣ Head Pose Estimation (PnP) — Distraction Detection
- Implements full 3D head pose estimation using:
  - `cv2.solvePnP()`
  - `cv2.Rodrigues()`
  - `cv2.RQDecomp3x3()`
- Extracts:
  - Pitch (up/down)
  - Yaw (left/right)
- Detects:
  - Looking away
  - Phone usage
  - Prolonged distraction

---

### 4️⃣ Temporal Intelligence — Sustained Distraction Logic
- Uses time-based persistence instead of instant alerts
- `SUSTAINED_DISTRACTION_TIME = 2.5` seconds
- Short glance → No alert  
- Continuous distraction → Alert

---

### 5️⃣ Microsleep Detection (Advanced Feature)
Microsleep is detected when:
- Eyes are closed
- Head is facing forward
- Closure duration exceeds a defined threshold

This mimics real physiological microsleep behavior.

---

### 6️⃣ Priority-Based Risk Assessment System
Driver state classification follows a hierarchical safety model:

1. Microsleep (highest risk)
2. Yawning
3. Excessive blinking
4. Sustained distraction
5. Attentive

Critical risks always override lower-priority states.

---

### 7️⃣ Real-Time Alerts with Cross-Platform Audio
- Windows → `winsound`
- macOS → `afplay`
- Linux → `paplay`

Ensures real-time feedback across operating systems.

---

### 8️⃣ Intelligent Event-Based Logging
- Logs only when the driver state changes
- Prevents redundant per-frame logging
- Produces clean, meaningful analytics data

---

## 🛠 Tech Stack
- Python
- OpenCV
- MediaPipe / Facial Landmarks
- NumPy
- Cross-platform OS utilities

---

## 🚀 Why Vertic AI?
- Combines computer vision, geometry, and temporal reasoning
- Avoids naive heuristics common in beginner projects
- Designed with real-world deployment constraints in mind
- Demonstrates safety-first and systems-level thinking

---

## 📌 Use Cases
- Driver Monitoring Systems (DMS)
- Fleet safety solutions
- Automotive ADAS research
- Real-time vision-based alertness monitoring

---

## 📄 License
MIT License
