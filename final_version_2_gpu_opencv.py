import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import time
from threading import Thread
from queue import Queue

# Load face detection model
face_model_file = "res10_300x300_ssd_iter_140000.caffemodel"
face_config_file = "deploy.prototxt"
face_net = cv2.dnn.readNetFromCaffe(face_config_file, face_model_file)
face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load emotion recognition model
emotion_model = load_model('EmotionRecTraining/model_out/')

# Load video
video_file = '/home/dl/Facial_Detection/3.mp4'
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()

# Get the width and height of the video to save the output video
ret, test_frame = cap.read()
h, w = test_frame.shape[:2]
cap.release()  # release camera

# Reopen the video file to restart reading
cap = cv2.VideoCapture(video_file)

# Set video frame rate and display delay
fps = 15
frame_delay = int(1000 / fps)

# Prepare sentiment labels and data storage
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_data = []

# Using a limited-size queue for asynchronous processing
frame_queue = Queue(maxsize=10)
emotion_queue = Queue(maxsize=10)

# Initialize video writing object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(os.getcwd(), 'output.mp4'), fourcc, fps, (w, h))

def process_frames():
    while True:
        frame = frame_queue.get()
        if frame is None:  # Check if it is an end signal
            emotion_queue.put(None)  # Send an end signal to emotion_queue
            break
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        faces = []  # Store detected faces
        coords = []  # Store face coordinates

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                faces.append(face)
                coords.append((startX, startY, endX, endY))
        
        if faces:
            faces_array = np.array(faces)
            predictions = emotion_model.predict(faces_array, batch_size=len(faces))
            predicted_emotions = [emotion_labels[idx] for idx in np.argmax(predictions, axis=1)]
            emotion_queue.put((frame, coords, predicted_emotions))
        else:
            emotion_queue.put((frame, [], []))

def display_frames():
    while True:
        item = emotion_queue.get()
        if item is None:  # Check if it is an end signal
            break
        frame, coords, predicted_emotions = item
        for i, (startX, startY, endX, endY) in enumerate(coords):
            emotion = predicted_emotions[i]
            cv2.putText(frame, emotion, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            emotion_data.append({
                "time": cap.get(cv2.CAP_PROP_POS_MSEC),
                "face_id": i,
                "emotion": emotion
            })
        cv2.imshow('Frame', frame)
        out.write(frame)  # Write frames to video file
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

# Start the thread that processes the frame
thread_process = Thread(target=process_frames, daemon=True)
thread_display = Thread(target=display_frames, daemon=True)
thread_process.start()
thread_display.start()

# main loop
while True:
    start_time = cv2.getTickCount()  # Get the current clock count
    ret, frame = cap.read()
    if not ret:
        frame_queue.put(None)  # When the video ends, put an end signal
        break
    if not frame_queue.full():
        frame_queue.put(frame)
    else:
        _ = frame_queue.get()  # discard oldest frame

    # Calculate how long it takes to process each frame and wait appropriately
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    time_to_wait = max(1 / fps - elapsed_time, 0)
    time.sleep(time_to_wait)

# Notification thread ends
emotion_queue.put(None)

# Wait for thread to end
thread_process.join()
thread_display.join()

# Clean and save data
cap.release()
out.release()  # Release the video write object
cv2.destroyAllWindows()
df = pd.DataFrame(emotion_data)
csv_file_path = os.path.join(os.getcwd(), 'emotion_data.csv')
df.to_csv(csv_file_path, index=False)
print(f"Emotion data has been saved to {csv_file_path}")
