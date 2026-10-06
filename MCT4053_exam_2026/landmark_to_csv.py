import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import pandas as pd
import os
from pathlib import Path

# this function is from a code-example at the website for the pose landmarker detection guide
def draw_landmarks_on_image(rgb_image, detection_result):
  annotated_image = np.copy(rgb_image)

  if not detection_result.pose_landmarks:
     return annotated_image
  
  pose_landmarks_list = detection_result.pose_landmarks
  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

model_path = 'pose_landmarker_lite.task'

base_options = mp.tasks.BaseOptions
pose_landmarker = mp.tasks.vision.PoseLandmarker
pose_landmarker_options = mp.tasks.vision.PoseLandmarkerOptions
vision_running_mode = mp.tasks.vision.RunningMode

video_root = Path('snutter')
output_dir = Path('lm_videoer')
output_dir.mkdir(parents=True, exist_ok=True)

all_rows = []

def process_video(input_path, video_name, move_name):
    joint_list = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    options = pose_landmarker_options(
        base_options=base_options(model_asset_path=model_path),
        running_mode=vision_running_mode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.4,
    )

    detector = pose_landmarker.create_from_options(options)

    print(f"Processing: {input_path}")
    video_id = input_path.stem

    cap = cv2.VideoCapture(str(input_path)) 
    if not cap.isOpened():
       print(f"Could not open videofile: {input_path}")
       detector.close()
       return

    fps = cap.get(cv2.CAP_PROP_FPS) 
    if not fps or fps <= 0:
        fps = 30

    # settings for making videos with landmarks and skeleton drawn onto them
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_dir = Path(f"{output_dir}/{move_name}/{video_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{video_id}_pose.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, 20, (frame_w, frame_h))

    timestamp_ms = 0
    frame_i = 0

    while True: 
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        result = detector.detect_for_video(mp_image, timestamp_ms)

        body_image = draw_landmarks_on_image(frame, result)
        out.write(body_image)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0] # for only one person

            for lm_i, lm_idx in enumerate(joint_list):
                lm = landmarks[lm_idx]
                all_rows.append({
                    'video': str(video_name),
                    'video_id': video_id,
                    'frame_idx':frame_i,
                    'timestamp_ms': timestamp_ms,
                    'landmark_index': lm_i,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    })
        
        frame_i += 1
        timestamp_ms += int(1000 / fps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Finished with: {input_path}")


for move_dir in video_root.iterdir():
    for video_dir in move_dir.iterdir():
        for input_path in sorted(video_dir.glob("*.mp4")):
            process_video(input_path, video_dir.name, move_dir.name)

df = pd.DataFrame(all_rows)
df.to_csv('landmarks_all_videos.csv', index=False)
print("Saved landmarks_all_videos.csv")
