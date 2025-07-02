import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
from collections import deque
from sklearn.model_selection import train_test_split
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Step 1: Collect keypoints from videos and images
class KeypointCollector:
    def __init__(self, sequence_length=10, max_people=5):
        self.sequence_length = sequence_length
        self.max_people = max_people
        self.keypoint_sequences = []
        self.current_sequences = [deque(maxlen=sequence_length) for _ in range(max_people)]

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Support multiple people (up to max_people)
            pose_landmarks_list = results.multi_pose_landmarks if hasattr(results, 'multi_pose_landmarks') else [results.pose_landmarks]
            num_people = min(len(pose_landmarks_list), self.max_people)

            for i in range(self.max_people):
                if i < num_people:
                    landmarks = [(lm.x, lm.y, lm.visibility) for lm in pose_landmarks_list[i].landmark]
                    self.current_sequences[i].append(landmarks)
                    if len(self.current_sequences[i]) == self.sequence_length:
                        self.keypoint_sequences.append(np.array(self.current_sequences[i], dtype=np.float32))
                else:
                    # Clear sequence for undetected people
                    self.current_sequences[i].clear()

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
                sequence = np.full((self.sequence_length, 33, 3), keypoints, dtype=np.float32)
                self.keypoint_sequences.append(sequence)
        self.current_sequences = [deque(maxlen=self.sequence_length) for _ in range(self.max_people)]

    def collect_sequences_from_folder(self, folder_path, label, is_video=True):
        sequences = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if is_video and filename.endswith((".mp4", ".avi")):
                print(f"Processing video: {file_path}...")
                self.collect_sequences_from_video(file_path)
                sequences.extend([(seq, label) for seq in self.keypoint_sequences])
                self.keypoint_sequences.clear()
            elif not is_video and filename.endswith((".jpg", ".jpeg", ".png")):
                print(f"Processing image: {file_path}...")
                self.collect_sequences_from_image(file_path)
                sequences.extend([(seq, label) for seq in self.keypoint_sequences])
                self.keypoint_sequences.clear()
        return sequences

# Step 2: Preprocess data for training
def preprocess_sequences(sequence_data):
    if not sequence_data:
        return None, None, None, None
    X = np.array([seq.reshape(-1, 33 * 3) for seq, _ in sequence_data], dtype=np.float32)  # [N, 10, 99]
    y = np.array([label for _, label in sequence_data], dtype=np.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Data augmentation (optional)
def augment_sequence(sequence):
    # Simple augmentation: random scaling and rotation
    scale = np.random.uniform(0.9, 1.1)
    angle = np.random.uniform(-15, 15) * np.pi / 180  # ±15 degrees
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    
    augmented = sequence.copy()
    for t in range(sequence.shape[0]):
        for i in range(33):
            x, y = augmented[t, i, 0], augmented[t, i, 1]
            augmented[t, i, 0] = (x - 0.5) * cos_val - (y - 0.5) * sin_val + 0.5  # Center, rotate, shift back
            augmented[t, i, 1] = (x - 0.5) * sin_val + (y - 0.5) * cos_val + 0.5
            augmented[t, i, :2] *= scale  # Apply scaling
    return augmented

# Step 4: Build a TFLite-friendly LSTM model
def build_lstm_model(seq_length=10, feature_size=99):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_length, feature_size), name='lstm_input'),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 5: Convert to TFLite
def convert_to_tflite(keras_model, output_path="violence_mediapipe_multiperson.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter.input_shapes = {'lstm_input': (1, 10, 99)}

    try:
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved as '{output_path}'")
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        raise

# Main function
def main():
    print("Starting training process...", flush=True)
    print(f"Using TensorFlow version: {tf.__version__}", flush=True)

    # Define folder paths
    video_violence_folder = "../../data/video/violence"
    video_non_violence_folder = "../../data/video/non_violence"
    image_violence_folder = "../../data/violence_dataset/violence"
    image_non_violence_folder = "../../data/violence_dataset/non_violence"

    # Create directories if they don’t exist
    for folder in [video_violence_folder, video_non_violence_folder, image_violence_folder, image_non_violence_folder]:
        os.makedirs(folder, exist_ok=True)

    # Collect sequences
    collector = KeypointCollector(sequence_length=10, max_people=5)
    
    print("Collecting video violent sequences...", flush=True)
    video_violent_sequences = collector.collect_sequences_from_folder(video_violence_folder, 1, is_video=True)
    
    print("Collecting video non-violent sequences...", flush=True)
    video_non_violent_sequences = collector.collect_sequences_from_folder(video_non_violence_folder, 0, is_video=True)
    
    # Uncomment for image data
    print("Collecting image violent sequences...", flush=True)
    image_violent_sequences = collector.collect_sequences_from_folder(image_violence_folder, 1, is_video=False)
    print("Collecting image non-violent sequences...", flush=True)
    image_non_violent_sequences = collector.collect_sequences_from_folder(image_non_violence_folder, 0, is_video=False)

    sequence_data = video_violent_sequences + video_non_violent_sequences # + image_violent_sequences + image_non_violent_sequences
    print(f"Total sequences collected: {len(sequence_data)}", flush=True)

    if not sequence_data:
        print("No sequences collected. Exiting.", flush=True)
        return

    # Preprocess and augment data
    print("Preprocessing data...", flush=True)
    X_train, X_test, y_train, y_test = preprocess_sequences(sequence_data)
    if X_train is None:
        print("Preprocessing failed. Exiting.", flush=True)
        return
    
    # Optional: Augment training data
    X_train_aug = np.concatenate([X_train, np.array([augment_sequence(seq.reshape(10, 33, 3)).reshape(-1, 99) for seq in X_train])])
    y_train_aug = np.concatenate([y_train, y_train])

    print(f"Training data shape: {X_train_aug.shape}, Test data shape: {X_test.shape}", flush=True)

    # Build and train the model
    print("Building and training LSTM model...", flush=True)
    lstm_model = build_lstm_model(seq_length=10, feature_size=99)
    lstm_model.fit(X_train_aug, y_train_aug, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate
    print("Evaluating model...", flush=True)
    loss, accuracy = lstm_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}", flush=True)

    # Save Keras model
    lstm_model.save("violence_mediapipe_multiperson.h5")
    print("Keras model saved as 'violence_mediapipe_multiperson.h5'", flush=True)

    # Convert to TFLite
    print("Converting to TFLite...", flush=True)
    convert_to_tflite(lstm_model)

if __name__ == "__main__":
    main()