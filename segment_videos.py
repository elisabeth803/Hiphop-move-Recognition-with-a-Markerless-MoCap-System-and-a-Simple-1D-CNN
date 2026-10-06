import cv2
import os
from pathlib import Path

video_root = Path("videoer")
i = 0

for move_dir in video_root.iterdir():
    if not move_dir.is_dir():
        continue
    
    output_dir = Path(f"snutter/{move_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in move_dir.iterdir():
        print()
        vid_dir = Path(f"snutter/{move_dir.name}/{input_path.name[:3]}")
        vid_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open the video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        seconds_per_clip = 4
        frames_per_clip = int(round(fps * seconds_per_clip))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        frame_in_clip = 0

        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # start new clip if we don't have one to write to
            if out is None:
                out_path = vid_dir / f"clip_{i}.mp4"
                out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                frame_in_clip = 0
                print(f"Starting clip {i} → {out_path}")
                i += 1

            out.write(frame)
            frame_in_clip += 1

            # if we have written enough frames for 4 seconds, finish the clip
            if frame_in_clip >= frames_per_clip:
                out.release()
                out = None
        # if frames in clip is less than frames_per_clip, close it
        if out is not None:
            out.release()
        
cap.release()
print("Done producing clips!")
