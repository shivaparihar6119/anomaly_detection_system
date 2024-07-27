import random
from collections import deque

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

model = tf.keras.models.load_model('Violence_Detection.h5')
# pickle_in = open('violence_detection.pkl', 'rb')
# model = pickle.load(pickle_in)
def show_pred_frames(processed_frames, SEQUENCE_LENGTH):

  plt.figure(figsize=(20, 15))

  frames_count = len(processed_frames)
  random_range = sorted(random.sample(range(SEQUENCE_LENGTH, frames_count), 12))

  for counter, random_index in enumerate(random_range, 1):
    plt.subplot(5, 4, counter)
    frame = processed_frames[random_index]  # Access frame from processed_frames list

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)

  plt.tight_layout()
  plt.show()
def predict_violence(video_file, model):

    cap = cv2.VideoCapture(video_file)
    IMAGE_HEIGHT=64
    IMAGE_WIDTH=64
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''
    frames = []
    if not cap.isOpened():
        print("Error opening video stream or file")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_queue.append(normalized_frame)
        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_labels_prob = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_prob)
            predicted_class_name = CLASSES_LIST[predicted_label]

        if predicted_class_name == "Violence":
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 12)
        else:
            cv2.putText(frame, predicted_class_name, (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 12)

        frames.append(frame)
        cv2.imshow('Violence Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()
    return frames


st.title("Violence Detection System")

uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])


CLASSES_LIST = ["NonViolence", "Violence"]
SEQUENCE_LENGTH = 15

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

        frames = predict_violence("temp_video.mp4", model)
        if frames is not None:
            show_pred_frames(frames, 12)
        else:
            print("Error processing video.")
    # else:
    #     st.error("Error decoding video.")

else:
    st.info("Upload a video file to start prediction.")
