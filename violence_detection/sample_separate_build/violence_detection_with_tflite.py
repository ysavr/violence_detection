import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from collections import deque
import time
import os  # For path checking

# Load MoveNet model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Define edges and colors for drawing
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}
COLORS = {'m': (255, 0, 255), 'c': (0, 255, 255), 'y': (0, 255, 0)}

class MoveNetViolenceDetector:
    def __init__(self, sequence_length=10, tflite_model_path="/Users/mobiledev/StudioProjects/violence-pose/violence_detection/sample_separate_build/violence_lstm_model_sample.tflite"):
        self.sequence_length = sequence_length
        self.frame_count = 0
        self.violence_frames = 0
        self.violence_positions = []
        self.current_sequences = [deque(maxlen=sequence_length) for _ in range(6)]
        self.start_time = None
        self.real_fps = 0.0
        
        # Check if TFLite file exists
        if not os.path.exists(tflite_model_path):
            print(f"Error: TFLite model file not found at: {tflite_model_path}")
            self.interpreter = None
            self.input_details = None
            self.output_details = None
            return
        
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("TFLite model loaded successfully from:", tflite_model_path)
            print("Input Details:", self.input_details)
            print("Output Details:", self.output_details)
        except Exception as e:
            print(f"Error loading TFLite model: {e}. No violence detection possible.")
            self.interpreter = None
            self.input_details = None
            self.output_details = None

    def draw_keypoints(self, frame, keypoints):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > 0.1:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

    def draw_connections(self, frame, keypoints):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for edge, color_key in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            if (c1 > 0.1) & (c2 > 0.1):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[color_key], 2)

    def predict_violence(self, keypoints_with_scores):
        if not self.interpreter:
            return False
        for i, person in enumerate(keypoints_with_scores):
            if np.max(person[:, 2]) > 0.1:
                self.current_sequences[i].append(person)
                if len(self.current_sequences[i]) == self.sequence_length:
                    seq = np.array(self.current_sequences[i]).reshape(1, self.sequence_length, 17 * 3).astype(np.float32)
                    try:
                        self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                        if prediction > 0.5:
                            return True
                    except RuntimeError as e:
                        print(f"Error during TFLite inference: {e}")
                        return False
        return False

    def process_frame(self, frame, frame_number, fps):
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        
        current_people = 0
        for person in keypoints_with_scores:
            if np.max(person[:, 2]) > 0.1:
                current_people += 1
                self.draw_connections(frame, person)
                self.draw_keypoints(frame, person)
        
        self.frame_count += 1
        if self.predict_violence(keypoints_with_scores):
            self.violence_frames += 1
            timestamp = frame_number / fps
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            self.violence_positions.append((frame_number, f"{minutes:02d}:{seconds:02d}"))
            cv2.putText(frame, "VIOLENCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.start_time is not None and self.frame_count > 1:
            elapsed = time.time() - self.start_time
            self.real_fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {self.real_fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame

    def detect_in_video(self, video_path=None):
        cap = cv2.VideoCapture(0 if video_path is None else video_path)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        self.start_time = time.time()
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame, frame_number, fps)
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
        
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        
        for person in keypoints_with_scores:
            if np.max(person[:, 2]) > 0.1:
                self.draw_connections(frame, person)
                self.draw_keypoints(frame, person)
        
        self.frame_count = 1
        if self.interpreter:
            for i, person in enumerate(keypoints_with_scores):
                if np.max(person[:, 2]) > 0.1:
                    sequence = [person] * self.sequence_length
                    seq = np.array(sequence).reshape(1, self.sequence_length, 17 * 3).astype(np.float32)
                    try:
                        self.interpreter.set_tensor(self.input_details[0]['index'], seq)
                        self.interpreter.invoke()
                        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                        if prediction > 0.5:
                            self.violence_frames = 1
                            cv2.putText(frame, "VIOLENCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break
                    except RuntimeError as e:
                        print(f"Error during TFLite inference: {e}")
                        break
        
        cv2.imshow('Violence Detection - Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print(f"\nSummary:")
        print(f"Total Frames: {self.frame_count}")
        print(f"Violence Frames: {self.violence_frames}")
        print(f"Violence Prediction: {'Yes' if self.violence_frames > 0 else 'No'}")

def main():
    print("Starting violence detection...")
    detector = MoveNetViolenceDetector(sequence_length=10, tflite_model_path="/Users/mobiledev/StudioProjects/violence-pose/violence_detection/sample_separate_build/violence_lstm_model_sample.tflite")
    
    # Test with video
    video_path = "/Users/mobiledev/StudioProjects/violence-pose/data/video/testing/testing.mp4"
    detector.detect_in_video(video_path)

    # Uncomment for image prediction
    # image_path = "/Users/mobiledev/StudioProjects/violence-pose/data/violence_dataset/non_violence/NV_938.mp4_frame4.jpg"
    # detector.detect_in_image(image_path)

if __name__ == "__main__":
    main()