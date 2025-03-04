import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
import os

# Load MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Step 1: Collect keypoints from videos
class KeypointCollector:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.keypoint_sequences = []
        self.current_sequences = [deque(maxlen=sequence_length) for _ in range(6)]

    def process_frame(self, frame):
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)
        results = movenet(input_img)
        keypoints = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        
        for i, person in enumerate(keypoints):
            if np.max(person[:, 2]) > 0.1:  # Confidence threshold
                self.current_sequences[i].append(person)
                if len(self.current_sequences[i]) == self.sequence_length:
                    self.keypoint_sequences.append(np.array(self.current_sequences[i]))

    def collect_sequences_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def collect_sequences_from_folder(self, folder_path, label):
        sequences = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".mp4", ".avi")):  # Add more formats if needed
                video_path = os.path.join(folder_path, filename)
                print(f"Processing {video_path}...")
                self.collect_sequences_from_video(video_path)
                sequences.extend([(seq, label) for seq in self.keypoint_sequences])
                self.keypoint_sequences = []  # Reset for next video
        return sequences

# Step 2: Prepare data for training
def preprocess_sequences(sequence_data):
    X = np.array([seq.reshape(-1, 17 * 3) for seq, _ in sequence_data])
    y = np.array([label for _, label in sequence_data])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build the LSTM model
def build_lstm_model(seq_length=10, feature_size=51):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(seq_length, feature_size), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to run everything
def main():
    print("Starting training process...")
    print(f"Using TensorFlow version: {tf.__version__}")

    # Define folder paths
    violence_folder = "../data/video/violence"
    non_violence_folder = "../data/video/non_violence"

    # Check if folders exist
    if not os.path.exists(violence_folder) or not os.path.exists(non_violence_folder):
        print("Error: One or both folders do not exist. Please create 'data/video/violence' and 'data/video/non_violence' with videos.")
        return

    # Collect sequences from folders
    collector = KeypointCollector(sequence_length=10)
    
    print("Collecting violent sequences...")
    violent_sequences = collector.collect_sequences_from_folder(violence_folder, 1)
    
    print("Collecting non-violent sequences...")
    non_violent_sequences = collector.collect_sequences_from_folder(non_violence_folder, 0)
    
    sequence_data = violent_sequences + non_violent_sequences
    print(f"Total sequences collected: {len(sequence_data)}")

    if not sequence_data:
        print("No sequences collected. Please check your video files.")
        return

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_sequences(sequence_data)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Build and train the model
    print("Building and training LSTM model...")
    lstm_model = build_lstm_model(seq_length=10, feature_size=51)
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = lstm_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the model
    lstm_model.save("violence_lstm_model.h5")
    print("Model saved as 'violence_lstm_model.h5'")

if __name__ == "__main__":
    main()