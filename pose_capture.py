import cv2
import mediapipe as mp
import numpy as np
import time
import os

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Drawing specs
FACE_CONNECTIONS_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
FACE_CONNECTIONS_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Init FaceMesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Output folder
output_dir = "./Dataset/raw/badac"
os.makedirs(output_dir, exist_ok=True)

# Pose capture tracking
captured_states = set()
last_pose = None
pose_start_time = None
CAPTURE_DELAY = 2  # seconds

# Pose classification function
def classify_pose(x_angle, y_angle):
    # Left / Right classification
    if y_angle < -15:
        return "Looking Left Extreme"
    elif y_angle < -10:
        return "Looking Left Moderate"
    elif y_angle < -5:
        return "Looking Left Mild"
    elif y_angle > 15:
        return "Looking Right Extreme"
    elif y_angle > 10:
        return "Looking Right Moderate"
    elif y_angle > 5:
        return "Looking Right Mild"
    # Up / Down classification
    elif x_angle > 15:
        return "Looking Up Extreme"
    elif x_angle > 10:
        return "Looking Up Moderate"
    elif x_angle > 5:
        return "Looking Up Mild"
    elif x_angle < -15:
        return "Looking Down Extreme"
    elif x_angle < -10:
        return "Looking Down Moderate"
    elif x_angle < -5:
        return "Looking Down Mild"
    # Center
    else:
        return "Looking Forward"

# Start camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Flip and convert
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    pose_label = "No face"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_3d, face_2d = [], []
            nose_2d, nose_3d = (0, 0), (0, 0, 0)

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, lm.z)

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            # Camera intrinsics
            focal_length = img_w
            cam_matrix = np.array([
                [focal_length, 0, img_w / 2],
                [0, focal_length, img_h / 2],
                [0, 0, 1]
            ])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix
            )
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x_angle, y_angle, z_angle = angles[0] * 360, angles[1] * 360, angles[2] * 360

            # Classify pose
            pose_label = classify_pose(x_angle, y_angle)

            # Capture if held for 2 seconds and not captured before
            current_time = time.time()
            if pose_label != "No face":
                if pose_label != last_pose:
                    last_pose = pose_label
                    pose_start_time = current_time
                else:
                    if (current_time - pose_start_time) >= CAPTURE_DELAY and pose_label not in captured_states:
                        timestamp = int(current_time)
                        filename = os.path.join(output_dir, f"auto1_{pose_label.replace(' ', '_')}_{timestamp}.jpg")
                        cv2.imwrite(filename, image)
                        print(f"[CAPTURED] {pose_label} → {filename}")
                        captured_states.add(pose_label)
            else:
                last_pose = None
                pose_start_time = None

            # Nose direction line
            nose_3d_array = np.array([nose_3d], dtype=np.float64)
            nose_2d_proj, _ = cv2.projectPoints(
                nose_3d_array, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d_proj[0][0][0]), int(nose_2d_proj[0][0][1]))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            # Show angles & pose
            cv2.putText(image, pose_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(image, f'x: {int(x_angle)} y: {int(y_angle)} z: {int(z_angle)}',
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Draw mesh
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FACE_CONNECTIONS_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=FACE_CONNECTIONS_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

    # Show window
    cv2.imshow('Head Pose Estimation', image)

    # Exit key
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
