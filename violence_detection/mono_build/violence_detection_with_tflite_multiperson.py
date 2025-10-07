import tensorflow as tf
import cv2
import numpy as np
from collections import deque
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define edges for drawing (MediaPipe Pose connections)
EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14),
    (14, 16), (11, 12), (11, 23), (12, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]
COLOR = (0, 255, 0)  # Green for all connections

class MediaPipeViolenceDetector:
    def __init__(self, sequence_length=10, tflite_model_path="violence_mediapipe_multiperson.tflite", max_people=5):
        self.sequence_length = sequence_length
        self.max_people = max_people
        self.frame_count = 0
        self.violence_frames = 0
        self.violence_positions = []  # (frame_number, timestamp) pairs
        self.current_sequences = [deque(maxlen=sequence_length) for _ in range(max_people)]
        self.start_time = None
        self.real_fps = 0.0

        # Initialize PoseLandmarker
        base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("TFLite model loaded successfully from:", tflite_model_path)
            print("Input Details:", self.input_details)  # Should be [1, 10, 99]
            print("Output Details:", self.output_details)
        except Exception as e:
            print(f"Error loading TFLite model: {e}. No violence detection possible.")
            self.interpreter = None

    def draw_keypoints(self, frame, landmarks):
        if not landmarks:
            return
        for landmark in landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            if landmark.presence > 0.1:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def draw_connections(self, frame, landmarks):
        if not landmarks:
            return
        shaped = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.presence) for lm in landmarks]
        for edge in EDGES:
            p1, p2 = edge
            x1, y1, c1 = shaped[p1]
            x2, y2, c2 = shaped[p2]
            if c1 > 0.1 and c2 > 0.1:
                cv2.line(frame, (x1, y1), (x2, y2), COLOR, 2)

    def predict_violence(self, pose_landmarks_list):
        if not self.interpreter or not pose_landmarks_list:
            return [False] * self.max_people

        predictions = [False] * self.max_people
        for i, landmarks in enumerate(pose_landmarks_list[:self.max_people]):
            keypoints = np.array([(lm.x, lm.y, lm.presence) for lm in landmarks], dtype=np.float32)
            self.current_sequences[i].append(keypoints)
            if len(self.current_sequences[i]) == self.sequence_length:
                seq = np.array(self.current_sequences[i], dtype=np.float32).reshape(1, self.sequence_length, 33 * 3)
                self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                predictions[i] = prediction > 0.5
        return predictions

    def process_frame(self, frame, frame_number, fps, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.landmarker.detect_for_video(mp_image, int(timestamp_ms))

        current_people = len(results.pose_landmarks) if results.pose_landmarks else 0
        for landmarks in results.pose_landmarks:
            self.draw_connections(frame, landmarks)
            self.draw_keypoints(frame, landmarks)

        self.frame_count += 1
        is_violent = self.predict_violence(results.pose_landmarks)
        if any(is_violent):
            self.violence_frames += 1
            timestamp = frame_number / fps
            minutes, seconds = divmod(int(timestamp), 60)
            self.violence_positions.append((frame_number, f"{minutes:02d}:{seconds:02d}"))
            cv2.putText(frame, "VIOLENCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if self.start_time and self.frame_count > 1:
            elapsed = time.time() - self.start_time
            self.real_fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {self.real_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        return frame

    def detect_in_video(self, video_path=None):
        cap = cv2.VideoCapture(0 if video_path is None else video_path)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.start_time = time.time()
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = frame_number * (1000 / fps)  # Convert frame to milliseconds
            processed_frame = self.process_frame(frame, frame_number, fps, timestamp_ms)
            cv2.imshow('Violence Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1

        elapsed_time = time.time() - self.start_time
        self.real_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nSummary:")
        print(f"Video FPS (Nominal): {fps}")
        print(f"Real-time FPS: {self.real_fps:.2f}")
        print(f"Total Frames: {self.frame_count}")
        print(f"Violence Frames: {self.violence_frames}")
        print(f"Violence Percentage: {round((self.violence_frames / self.frame_count) * 100, 2) if self.frame_count > 0 else 0}%")
        if self.violence_positions:
            print("Violence Detected at:")
            for frame_num, timestamp in self.violence_positions:
                print(f"  Frame {frame_num} (Time: {timestamp})")

    def detect_in_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = self.landmarker.detect(mp_image)  # Single image detection

        for landmarks in results.pose_landmarks:
            self.draw_connections(frame, landmarks)
            self.draw_keypoints(frame, landmarks)

        self.frame_count = 1
        is_violent = self.predict_violence(results.pose_landmarks)
        if any(is_violent):
            self.violence_frames = 1
            cv2.putText(frame, "VIOLENCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Violence Detection - Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"\nSummary:")
        print(f"Total Frames: {self.frame_count}")
        print(f"Violence Frames: {self.violence_frames}")
        print(f"Violence Prediction: {'Yes' if self.violence_frames > 0 else 'No'}")

def main():
    print("Starting violence detection...")
    detector = MediaPipeViolenceDetector(sequence_length=10, tflite_model_path="violence_mediapipe_multiperson.tflite")
    
    # Test with video
    video_path = "../../data/video/testing/bullying_siswa.mp4"
    detector.detect_in_video(video_path)

    # Test with image (uncomment as needed)
    # image_path = "../data/violence_dataset/non_violence/NV_938.mp4_frame4.jpg"
    # detector.detect_in_image(image_path)

if __name__ == "__main__":
    main()