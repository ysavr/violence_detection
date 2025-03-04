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

# Step 1: Collect keypoints from videos and images
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
            if np.max(person[:, 2]) > 0.1:
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

    def collect_sequences_from_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load {image_path}")
            return
        self.process_frame(frame)
        for i, seq in enumerate(self.current_sequences):
            if seq:
                keypoints = seq[-1]
                sequence = np.array([keypoints] * self.sequence_length)
                self.keypoint_sequences.append(sequence)
        self.current_sequences = [deque(maxlen=self.sequence_length) for _ in range(6)]

    def collect_sequences_from_folder(self, folder_path, label, is_video=True):
        sequences = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if is_video and filename.endswith((".mp4", ".avi")):
                print(f"Processing video: {file_path}...")
                self.collect_sequences_from_video(file_path)
                sequences.extend([(seq, label) for seq in self.keypoint_sequences])
                self.keypoint_sequences = []
            elif not is_video and filename.endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing image: {file_path}...")
                self.collect_sequences_from_image(file_path)
                sequences.extend([(seq, label) for seq in self.keypoint_sequences])
                self.keypoint_sequences = []
        return sequences

# Step 2: Prepare data for training
def preprocess_sequences(sequence_data):
    X = np.array([seq.reshape(-1, 17 * 3) for seq, _ in sequence_data])
    y = np.array([label for _, label in sequence_data])
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build a TFLite-friendly LSTM model with explicit Input layer
def build_lstm_model(seq_length=10, feature_size=51):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, feature_size), name='lstm_input'),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')  # Named output for clarity
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function
def main():
    print("Starting training process...")
    print(f"Using TensorFlow version: {tf.__version__}")

    # Define folder paths
    video_violence_folder = "../../data/video/violence"
    video_non_violence_folder = "../../data/video/non_violence"
    image_violence_folder = "../../data/violence_dataset/violence"
    image_non_violence_folder = "../../data/violence_dataset/non_violence"

    # Check if folders exist
    for folder in [video_violence_folder, video_non_violence_folder, image_violence_folder, image_non_violence_folder]:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist. Creating it.")
            os.makedirs(folder)

    # Create a collector
    collector = KeypointCollector(sequence_length=10)
    
    # Collect sequences from videos
    print("Collecting video violent sequences...")
    video_violent_sequences = collector.collect_sequences_from_folder(video_violence_folder, 1, is_video=True)
    
    print("Collecting video non-violent sequences...")
    video_non_violent_sequences = collector.collect_sequences_from_folder(video_non_violence_folder, 0, is_video=True)
    
    # Collect sequences from images
    print("Collecting image violent sequences...")
    image_violent_sequences = collector.collect_sequences_from_folder(image_violence_folder, 1, is_video=False)
    
    print("Collecting image non-violent sequences...")
    image_non_violent_sequences = collector.collect_sequences_from_folder(image_non_violence_folder, 0, is_video=False)
    
    # Combine all sequences
    sequence_data = video_violent_sequences + video_non_violent_sequences + image_violent_sequences + image_non_violent_sequences
    print(f"Total sequences collected: {len(sequence_data)}")

    if not sequence_data:
        print("No sequences collected. Please check your video and image files.")
        return

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_sequences(sequence_data)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # Build and train the model
    print("Building and training LSTM model...")
    lstm_model = build_lstm_model(seq_length=10, feature_size=51)
    dummy_input = np.zeros((1, 10, 51), dtype=np.float32)
    lstm_model(dummy_input)  # Ensure input is defined before training
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the model
    print("Evaluating model...")
    loss, accuracy = lstm_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Save the model in SavedModel format
    saved_model_dir = "violence_lstm_saved_model"
    tf.saved_model.save(lstm_model, saved_model_dir)
    print(f"Model saved in SavedModel format at: {saved_model_dir}")

if __name__ == "__main__":
    main()