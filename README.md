# Hip Hop Recognition Program

The structure of the project directory: 
```text
MCT4053_exam_2026
├── snutter/                    # segmented versions of the videos
├── videoer/                    # raw, unprocessed videoes
├── cnn_simple_norm_params.npz  # parameters such as mean and standard deviation for the model
├── cnn_simple.h5               # ready-to-use model-file
├── csv_to_dataset.py           
├── landmark_to_csv.py
├── landmarks_all_videos.csv
├── mocap_dataset.npz
├── pose_landmarker_lite.task
├── README.md
├── realtime_classification.py
├── requirements.txt
├── segment_videos.py
├── test.py
├── train_simple_cnn.py
└── trim_videos.py
``` 
The program was written in Python, and the the libraries OpenCV, MediaPipe, Numpy, Tensorflow, Scikit-learn and Matplotlib were used. 
There are in total 7 python-scripts that are runnable. 
There is no need to run trim_videos.py. The videos in the directory "snutter" are already trimmed videos of the videos from the directory "videoer". 
Creating a virtual python environment is recommended. 
Activate the virtual environment. 

Run 
```bash 
pip install -r requirements.txt
```
There is a trained model in the project directory. If you only want to utilize this, jump to step 6. below. 
Otherwise, if you want to build the dataset from scratch, start from step 1. 
To only retrain the model, start from step 4. 
To recreate the results from scratch you can run all the programs listed under. This is the order needed to run the program:

## 1. segment_videos.py
The first program to be run is segment_videos.py.
Run 
```bash 
python3 segment_videos.py
```

## 2. landmark_to_csv.py
Next run 
```bash 
python3 landmark_to_csv.py
```

## 3. csv_to_dataset.py
Next run 
```bash 
python3 csv_to_dataset.py
```

## 4. train_simple_cnn.py
To train the model run 
```bash 
python3 train_simple_cnn.py
``` 

## 5. test.py
To test the model you can run 
```bash 
python3 test.py
```
## 6. realtime_classification.py 
To run the model in realtime run 
```bash 
python3 realtime_classification.py
```
This requires that you have a webcam available. You might have to change the line 69 of realtime_classification.py if you cannot find the webcam. Perhaps try 1 instead of 0.