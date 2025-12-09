import os
import csv
import time
import platform
import numpy as np
import cv2
import mediapipe as mp
from datetime import datetime

# Try importing winsound (Windows only)
try:
    import winsound
except Exception:
    winsound = None

# ============================
# CONFIGURATION / THRESHOLDS
# ============================

previous_distraction_status = "None"
previous_prediction = "None"

EAR_THRESH = 0.21  # Below → blink
EAR_CONSEC_FRAMES = 3  # Frames required to confirm blink
MAR_THRESH = 0.6  # Above → yawn
YAWN_CONSEC_FRAMES = 10  # Frames required to confirm yawn
ALERT_DEBOUNCE_SEC = 2.0  # Delay between repeated alerts
LOG_CSV = "results_phase3_log.csv"

# Head pose thresholds (REDUCED for earlier detection)
HEAD_POSE_DISTRACTION_YAW = 20  # Degrees left/right before considered distracted
HEAD_POSE_DISTRACTION_PITCH_DOWN = 15  # Degrees down before considered distracted
HEAD_POSE_DISTRACTION_PITCH_UP = 15  # Degrees up before considered distracted

# Sustained distraction threshold
SUSTAINED_DISTRACTION_TIME = 2.5  # Seconds of continuous distraction before alert

# Microsleep detection thresholds
MICROSLEEP_EAR_THRESH = 0.21  # Below this = eyes closed
MICROSLEEP_TIME_THRESH = 1.5  # Seconds with eyes closed = microsleep alert
MICROSLEEP_CONSEC_FRAMES = 2  # Frames to confirm eyes are closed

# MediaPipe Landmark Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

MOUTH = {
    "left_corner": 61,
    "right_corner": 291,
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "upper_inner": 81,
    "lower_inner": 311,
}

# ============================
# UTILITY FUNCTIONS
# ============================


def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    A = euclid(pts[1], pts[5])
    B = euclid(pts[2], pts[4])
    C = euclid(pts[0], pts[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(landmarks, img_w, img_h):
    lc = landmarks[MOUTH["left_corner"]]
    rc = landmarks[MOUTH["right_corner"]]

    ut_lm = landmarks[MOUTH["upper_lip_top"]]
    lt_lm = landmarks[MOUTH["lower_lip_bottom"]]

    lc_pt = (lc.x * img_w, lc.y * img_h)
    rc_pt = (rc.x * img_w, rc.y * img_h)
    ut_pt = (ut_lm.x * img_w, ut_lm.y * img_h)
    lt_pt = (lt_lm.x * img_w, lt_lm.y * img_h)

    vertical = euclid(ut_pt, lt_pt)
    width = euclid(lc_pt, rc_pt)

    if width == 0:
        return 0.0

    return vertical / width


def try_beep():
    """Cross-platform beep alert."""
    sysname = platform.system()
    try:
        if sysname == "Windows" and winsound:
            winsound.Beep(1000, 180)
        elif sysname == "Darwin":
            os.system("afplay /System/Library/Sounds/Glass.aiff >/dev/null 2>&1 &")
        else:
            os.system(
                "paplay /usr/share/sounds/freedesktop/stereo/complete.oga >/dev/null 2>&1 &"
            )
            print("\a", end="", flush=True)
    except Exception:
        pass  # Fail silently if sound not available


# ============================
# SETUP MEDIAPIPE + CAMERA + CSV
# ============================

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

# CSV Logging
csv_exists = os.path.exists(LOG_CSV)
log_file = open(LOG_CSV, "a", newline="")
writer = csv.writer(log_file)

if not csv_exists:
    writer.writerow(
        [
            "Timestamp",
            "HeadPose",
            "Prediction",
            "EAR_left",
            "EAR_right",
            "MAR",
            "FPS",
            "EventType",
            "DistractionStatus",
        ]
    )

# ============================
# INTERNAL STATE VARIABLES
# ============================

blink_counter_left = 0
blink_counter_right = 0
yawn_counter = 0
last_alert_time = 0.0

# Distraction tracking
distraction_start_time = None
is_currently_distracted = False

# NEW: Microsleep tracking
microsleep_start_time = None
is_microsleep_detected = False
microsleep_frame_counter = 0

# ============================
# MAIN PROCESSING LOOP
# ============================

with (
    mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh,
    mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection,
):
    while cap.isOpened():
        start_time = time.time()
        success, image = cap.read()

        if not success:
            continue

        # Flip horizontally for natural view
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        ih, iw, _ = image.shape

        detection_results = face_detection.process(image)
        mesh_results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        head_text = "NoFace"
        prediction_text = "NoFace"
        event_type = ""
        distraction_status = "None"

        ear_left = 0.0
        ear_right = 0.0
        mar = 0.0

        pitch_angle = 0.0
        yaw_angle = 0.0

        # Draw detection bounding box
        if detection_results.detections:
            for det in detection_results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Process face landmarks
        if mesh_results.multi_face_landmarks:
            face_landmarks = mesh_results.multi_face_landmarks[0]
            mp_drawing.draw_landmarks(
                image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
            )

            lm = face_landmarks.landmark

            # ============================
            # HEAD POSE ESTIMATION
            # ============================
            face_2d = []
            face_3d = []

            for idx in [33, 263, 1, 61, 291, 199]:
                x = int(lm[idx].x * iw)
                y = int(lm[idx].y * ih)
                face_2d.append([x, y])
                face_3d.append([x, y, lm[idx].z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = iw
            cam_matrix = np.array(
                [[focal_length, 0, ih / 2], [0, focal_length, iw / 2], [0, 0, 1]]
            )

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            try:
                ok, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                pitch_angle = angles[0] * 360  # X-axis (up/down)
                yaw_angle = angles[1] * 360  # Y-axis (left/right)
                z_ang = angles[2] * 360

                # Determine head position
                if yaw_angle < -10:
                    head_text = "Looking Left"
                elif yaw_angle > 10:
                    head_text = "Looking Right"
                elif pitch_angle < -10:
                    head_text = "Looking Down"
                elif pitch_angle > 10:
                    head_text = "Looking Up"
                else:
                    head_text = "Forward"

                # Check if distracted based on thresholds
                is_head_distracted = False

                if yaw_angle < -HEAD_POSE_DISTRACTION_YAW:
                    is_head_distracted = True
                    distraction_status = "Looking Far Left"
                elif yaw_angle > HEAD_POSE_DISTRACTION_YAW:
                    is_head_distracted = True
                    distraction_status = "Looking Far Right"
                elif pitch_angle < -HEAD_POSE_DISTRACTION_PITCH_DOWN:
                    is_head_distracted = True
                    distraction_status = "Looking Down (Phone?)"
                elif pitch_angle > HEAD_POSE_DISTRACTION_PITCH_UP:
                    is_head_distracted = True
                    distraction_status = "Looking Up"
                else:
                    is_head_distracted = False
                    distraction_status = "Attentive"

                # Track sustained distraction
                now = time.time()

                if is_head_distracted:
                    if distraction_start_time is None:
                        distraction_start_time = now

                    distraction_duration = now - distraction_start_time

                    # If distracted for required time, mark as currently distracted
                    if distraction_duration >= SUSTAINED_DISTRACTION_TIME:
                        is_currently_distracted = True
                else:
                    # Reset distraction tracking when looking forward
                    distraction_start_time = None
                    is_currently_distracted = False

            except Exception:
                head_text = "PoseErr"
                distraction_status = "Error"
                is_currently_distracted = False
                distraction_start_time = None

            # ============================
            # EAR / MAR COMPUTATION
            # ============================

            ear_left = eye_aspect_ratio(lm, LEFT_EYE, iw, ih)
            ear_right = eye_aspect_ratio(lm, RIGHT_EYE, iw, ih)
            mar = mouth_aspect_ratio(lm, iw, ih)

            # ============================
            # MICROSLEEP DETECTION
            # ============================

            now = time.time()
            avg_ear = (ear_left + ear_right) / 2.0

            # Check if eyes are closed while looking forward
            is_looking_forward = abs(yaw_angle) < 10 and abs(pitch_angle) < 10

            if avg_ear < MICROSLEEP_EAR_THRESH and is_looking_forward:
                microsleep_frame_counter += 1

                # Start timer after confirming eyes are closed for consecutive frames
                if microsleep_frame_counter >= MICROSLEEP_CONSEC_FRAMES:
                    if microsleep_start_time is None:
                        microsleep_start_time = now

                    microsleep_duration = now - microsleep_start_time

                    # Check if microsleep threshold reached
                    if microsleep_duration >= MICROSLEEP_TIME_THRESH:
                        is_microsleep_detected = True
                    else:
                        # Show countdown
                        remaining = MICROSLEEP_TIME_THRESH - microsleep_duration
                        cv2.putText(
                            image,
                            f"Eyes Closed! Alert in {remaining:.1f}s",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 165, 255),
                            2,
                        )
            else:
                # Reset microsleep tracking
                microsleep_frame_counter = 0
                microsleep_start_time = None
                is_microsleep_detected = False

            # ============================
            # BLINK DETECTION
            # ============================

            if ear_left < EAR_THRESH:
                blink_counter_left += 1
            else:
                if blink_counter_left >= EAR_CONSEC_FRAMES:
                    event_type = "Blink_Left"
                blink_counter_left = 0

            if ear_right < EAR_THRESH:
                blink_counter_right += 1
            else:
                if blink_counter_right >= EAR_CONSEC_FRAMES:
                    event_type = (
                        "Blink_Right" if not event_type else event_type + ";Blink_Right"
                    )
                blink_counter_right = 0

            # ============================
            # YAWN DETECTION (FIXED)
            # ============================

            if mar > MAR_THRESH:
                yawn_counter += 1
                if yawn_counter >= YAWN_CONSEC_FRAMES:
                    # Add yawn event immediately when threshold reached
                    if "Yawn" not in event_type:
                        event_type = "Yawn" if not event_type else event_type + ";Yawn"
            else:
                yawn_counter = 0

            # ============================
            # FINAL PREDICTION LOGIC
            # ============================

            # Priority 1: Microsleep (HIGHEST PRIORITY)
            if is_microsleep_detected:
                prediction_text = "MICROSLEEP DETECTED!"
                distraction_status = "Microsleep"
                if "Microsleep" not in event_type:
                    event_type = (
                        "Microsleep" if not event_type else event_type + ";Microsleep"
                    )

            # Priority 2: Yawning (drowsiness)
            elif "Yawn" in event_type and (now - last_alert_time > ALERT_DEBOUNCE_SEC):
                prediction_text = "DROWSY - Yawning"

            # Priority 3: Excessive blinking (drowsiness)
            elif "Blink" in event_type and (now - last_alert_time > ALERT_DEBOUNCE_SEC):
                prediction_text = "Excessive Blinking"

            # Priority 4: Sustained distraction
            elif is_currently_distracted:
                prediction_text = f"DISTRACTED - {distraction_status}"
                if "Distraction" not in event_type:
                    event_type = (
                        "Distraction" if not event_type else event_type + ";Distraction"
                    )

            # Normal state
            else:
                prediction_text = f"Attentive - {head_text}"
                if distraction_start_time is not None:
                    # Show warning that distraction is being tracked
                    elapsed = now - distraction_start_time
                    remaining = SUSTAINED_DISTRACTION_TIME - elapsed
                    prediction_text += f" (Alert in {remaining:.1f}s)"

            # Draw EAR / MAR
            cv2.putText(
                image,
                f"EAR L:{ear_left:.2f} R:{ear_right:.2f}",
                (10, ih - 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                image,
                f"MAR:{mar:.2f}",
                (10, ih - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            # Display head angles for debugging
            cv2.putText(
                image,
                f"Pitch:{pitch_angle:.1f} Yaw:{yaw_angle:.1f}",
                (10, ih - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

            # Alert Display
            if event_type and (now - last_alert_time) > ALERT_DEBOUNCE_SEC:
                # Alert for drowsiness, sustained distraction, or microsleep
                should_alert = (
                    "Yawn" in event_type
                    or "Blink" in event_type
                    or "Distraction" in event_type
                    or "Microsleep" in event_type
                )

                if should_alert:
                    cv2.rectangle(image, (0, 0), (iw, 40), (0, 0, 255), -1)
                    cv2.putText(
                        image,
                        f"ALERT: {event_type}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                    try_beep()
                    last_alert_time = now

        else:
            # No face detected - reset all tracking
            distraction_start_time = None
            is_currently_distracted = False
            microsleep_start_time = None
            is_microsleep_detected = False
            microsleep_frame_counter = 0

        # ============================
        # FPS DISPLAY
        # ============================

        end_time = time.time()
        fps = 1.0 / max((end_time - start_time), 1e-5)

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        # Prediction text - color coded
        text_color = (0, 255, 0)  # Green for attentive
        if (
            "DISTRACTED" in prediction_text
            or "DROWSY" in prediction_text
            or "MICROSLEEP" in prediction_text
        ):
            text_color = (0, 0, 255)  # Red for alerts
        elif "Alert in" in prediction_text or "Eyes Closed" in prediction_text:
            text_color = (0, 165, 255)  # Orange for warning

        cv2.putText(
            image,
            f"Status: {prediction_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
        )

        # Show window
        cv2.imshow("Distraction Detection Phase-3", image)

        # ============================
        # LOGGING - Only log on status change
        # ============================

        # Determine if we should log this frame
        should_log = False

        # Check if distraction status changed
        if distraction_status != previous_distraction_status:
            should_log = True
            previous_distraction_status = distraction_status

        # Check if prediction changed (for alerts like Yawn, Blink, Microsleep)
        if prediction_text != previous_prediction:
            should_log = True
            previous_prediction = prediction_text

        # Only write to CSV if there's a change
        if should_log:
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    head_text,
                    prediction_text,
                    f"{ear_left:.3f}",
                    f"{ear_right:.3f}",
                    f"{mar:.3f}",
                    f"{fps:.2f}",
                    event_type,
                    distraction_status,
                ]
            )
            log_file.flush()

        # Exit key (FIXED - now properly detects 'q')
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

# ============================
# CLEANUP
# ============================

cap.release()
cv2.destroyAllWindows()
log_file.close()
print("\nApplication closed successfully!")
print(f"Logs saved to: {LOG_CSV}")
