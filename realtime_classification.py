import numpy as np
from tensorflow import keras
from collections import deque
import cv2
import mediapipe as mp

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode

# Load model
model = keras.models.load_model("cnn_simple.h5")

# Load normalization parameters
norm = np.load("cnn_simple_norm_params.npz")
mean = norm["mean"]   # shape (1, 1, 48)
std  = norm["std"]    # shape (1, 1, 48)

T = 90     # sequence length used in training 
NUM_DIMS = 3

def preprocess_window(window_xyz):
    # window_xyz: (T, 16, 3) -> (1, T, 48) normalized
    X = window_xyz.reshape(T, NUM_JOINTS * NUM_DIMS)  # (T, 48)
    X_norm = (X - mean.squeeze()) / std.squeeze()
    return X_norm[np.newaxis, :, :]  # (1, T, 48)


def predict_from_buffer(f_buffer):
    # use the current frame_buffer to run the CNN.

    if len(f_buffer) < T: # returns (pred_class, confidence) or None if not enough frames.
        return None

    window_xyz = np.stack(f_buffer, axis=0)  # (T, 16, 3)
    X_input = preprocess_window(window_xyz)
    probs = model.predict(X_input)[0]
    pred = int(np.argmax(probs))
    conf = probs[pred]

    return pred, conf


model_path = 'pose_landmarker_lite.task'

# same as when pre-processing the data
options = pose_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

detector = pose_landmarker.create_from_options(options)

# map class indices to names
CLASS_NAMES = {
    0: "Bart Simpson",
    1: "Cabbage Patch",
    2: "Kick Ball-Change",
    3: "Pas de Bourre",
    4: "Steve Martin",
    5: "Wu-Tang",
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")

# try to get fps from camera or default to 30
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30

timestamp_ms = 0
frame_buffer = deque(maxlen=T)

joint_list = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
NUM_JOINTS = len(joint_list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # wrap frame in MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # video mode: synchronous call with timestamp
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        joints = np.zeros((NUM_JOINTS, NUM_DIMS))
        for j, lm_idx in enumerate(joint_list):
            lm = landmarks[lm_idx]
            joints[j, 0] = lm.x
            joints[j, 1] = lm.y
            joints[j, 2] = lm.z

        # normalizing like offline
        # mid-hip
        mid_hip = 0.5 * (joints[6] + joints[7])   # (3,)
        joints_centered = joints - mid_hip        # (16, 3)

        # shoulder-width
        shoulder_dist = np.linalg.norm(joints_centered[0] - joints_centered[1]) + 1e-6
        joints_norm = joints_centered / shoulder_dist

        frame_buffer.append(joints_norm)

        # predict if enough frames
        pred = predict_from_buffer(frame_buffer)
        if pred is not None:
            pred_class, conf = pred
            class_name = CLASS_NAMES.get(pred_class, str(pred_class))
            pred_text = f"{class_name}  ({conf:.2f})"
        else:
            pred_text = "Collecting..."

    else:
        pred_text = "No pose detected"

    # text in video
    cv2.putText(frame, f"Pred: {pred_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Hiphop-Move Classification", frame)

    # increase timestamp
    timestamp_ms += int(1000 / fps)

    # avslutt programmet med 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
