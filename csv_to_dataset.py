import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

NUM_DIMS = 3  # x, y, z
T = 90  # sequence length
step = 30  # 1/3 overlap between the sequences
joint_list = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
NUM_JOINTS = len(joint_list)

df = pd.read_csv("landmarks_all_videos.csv")
videos = sorted(df["video"].unique())

video_labels = []

for vid in videos:
    if vid < 100:
        label = 0
    else:
        label = int(str(vid)[0])
    video_labels.append(label)
    
videos = np.array(videos)
video_labels = np.array(video_labels)

vid_trainval, vid_test, y_vid_trainval, y_vid_test = train_test_split(
    videos,
    video_labels,
    test_size=0.2,
    stratify=video_labels,
    random_state=42
)

vid_train, vid_val, y_vid_train, y_vid_val = train_test_split(
    vid_trainval,
    y_vid_trainval,
    test_size=0.25,
    stratify=y_vid_trainval,
    random_state=42
)

def build_windows_for_videos(video_list, labels, df):
    X_list = []
    y_list = []

    # process one video/move at a time
    for vid, label in zip(video_list, labels):

        df_video = df[df["video"] == vid].copy()
        df_video.sort_values(["video_id"], inplace=True)
        
        video_ids = sorted(df_video["video_id"].unique())
        for vid_id in video_ids:
            df_clip = df[df["video_id"] == vid_id].copy()

            # sort after frame and landmark_index
            df_clip.sort_values(["frame_idx", "landmark_index"], inplace=True)

            frame_indices = sorted(df_clip["frame_idx"].unique())
            num_frames = len(frame_indices)

            if num_frames < T:
                print(f"Video {vid_id} has {num_frames} frames. Skipping.")
                continue

            # making an array with the shape (num_frames, NUM_JOINTS, NUM_DIMS)
            frames = np.full((num_frames, NUM_JOINTS, NUM_DIMS), np.nan)

            for i, frame_i in enumerate(frame_indices):
                df_f = df_clip[df_clip["frame_idx"] == frame_i].sort_values("landmark_index")
                coords = df_f[["x", "y", "z"]].values  # (16, 3)
                idxs = range(NUM_JOINTS)
                frames[i, idxs, :] = coords[:NUM_JOINTS, :]

            time = np.arange(num_frames)
            
            # interpolation-loop for when landmarks are missing
            for lm_i in range(len(joint_list)):
                for d in range(NUM_DIMS): 
                    series = frames[:, lm_i, d]
                    mask = ~np.isnan(series)

                    if mask.sum() == 0:
                        frames[:, lm_i, d] = 0
                    elif mask.sum() == 1:
                        val = series[mask][0]
                        frames [:, lm_i, d] = val
                    else:
                        frames[~mask, lm_i, d] = np.interp(
                            time[~mask],
                            time[mask],
                            series[mask]
                        )

            # normalize the skeleton: centralize + scale per frame
            # mid-hip in mediapipe: 23 = left hip, 24 = right hip. in our joint_list it corresponds to indexes 6 and 7, respectively. 
            mid_hip = 0.5 * (frames[:, 6, :] + frames[:, 7, :])   # (num_frames, 3)
            frames = frames - mid_hip[:, np.newaxis, :] # (num_frames, 1, 3) move the whole body to origo, numpy broadcasting ensures that mid_hip is subtracted from all joint-values

            # scale shoulder-width: 11 = left shoulder, 12 = right shoulder. in our joint_list it corresponds to indexes 0 and 1, respectively. 
            shoulder_dist = np.linalg.norm(
                frames[:, 0, :] - frames[:, 1, :],
                axis=-1,
                keepdims=True
            )  # (num_frames, 1)
            scale = shoulder_dist + 1e-6 # adding a small number to avoid division by 0
            frames = frames / scale[:, np.newaxis] # (num_frames, NUM_JOINTS, NUM_DIMS)

            samples = []
            start_indices = []

            for start in range(0, num_frames - T + 1, step):
                end = start + T
                window = frames[start:end]  # (T, 16, 3)
                samples.append(window)
                start_indices.append(start)

            samples = np.stack(samples, axis=0)  # (N, T, 16, 3)
            N = samples.shape[0]

            X = samples.reshape(N, T, NUM_JOINTS * NUM_DIMS)  # (N, T, 48)
            y = np.full((N,), label)
                
            X_list.append(X)
            y_list.append(y)

    X_concat = np.concatenate(X_list, axis=0) # (total_N, T, 48)
    y_concat = np.concatenate(y_list, axis=0) # (total_N)
    
    return X_concat, y_concat


X_train, y_train = build_windows_for_videos(vid_train, y_vid_train, df)
X_val, y_val = build_windows_for_videos(vid_val, y_vid_val, df)
X_test, y_test = build_windows_for_videos(vid_test, y_vid_test, df)

# save to disk for training and testing
np.savez_compressed(
    "mocap_dataset.npz",
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
    X_test=X_test,   y_test=y_test,
)