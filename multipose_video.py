import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from collections import defaultdict
import time

# Load MoveNet MultiPose Lightning model
model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Define edges for drawing connections
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}
COLORS = {'m': (255, 0, 255), 'c': (0, 255, 255), 'y': (0, 255, 0)}

class MoveNetSummarizer:
    def __init__(self, confidence_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.detection_frames = 0
        self.max_people = 0
        self.total_people_detected = 0
        self.start_time = 0
        self.total_duration = 0
        self.pose_stats = defaultdict(int)
        self.violence_stats = defaultdict(int)  # Track violent actions
        self.prev_keypoints = None  # For velocity tracking
        self.violence_detected = False

    def draw_keypoints(self, frame, keypoints):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > self.confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

    def draw_connections(self, frame, keypoints):
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for edge, color_key in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            if (c1 > self.confidence_threshold) & (c2 > self.confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[color_key], 2)

    def analyze_violence(self, keypoints_with_scores, frame_time):
        violent_frames = 0
        current_keypoints = keypoints_with_scores.copy()

        for i, person in enumerate(current_keypoints):
            if np.max(person[:, 2]) < self.confidence_threshold:
                continue
            
            shaped = np.squeeze(person)
            nose = shaped[0]
            left_shoulder = shaped[5]
            right_shoulder = shaped[6]
            left_elbow = shaped[7]
            right_elbow = shaped[8]
            left_wrist = shaped[9]
            right_wrist = shaped[10]
            left_hip = shaped[11]
            right_hip = shaped[12]
            left_knee = shaped[13]
            right_knee = shaped[14]

            # Violence indicators
            # 1. Raised arms (punching motion)
            shoulder_level = (left_shoulder[0] + right_shoulder[0]) / 2
            if (left_wrist[2] > self.confidence_threshold and left_wrist[0] < shoulder_level - 0.1) or \
               (right_wrist[2] > self.confidence_threshold and right_wrist[0] < shoulder_level - 0.1):
                self.violence_stats['raised_arms'] += 1
                violent_frames += 1

            # 2. Kicking motion
            hip_level = (left_hip[0] + right_hip[0]) / 2
            if (left_knee[2] > self.confidence_threshold and left_knee[0] < hip_level - 0.15) or \
               (right_knee[2] > self.confidence_threshold and right_knee[0] < hip_level - 0.15):
                self.violence_stats['kicking'] += 1
                violent_frames += 1

            # 3. Falling/lying on ground
            if nose[2] > self.confidence_threshold and nose[0] > 0.8 and hip_level > 0.7:
                self.violence_stats['falling'] += 1
                violent_frames += 1

            # 4. Rapid movement (velocity)
            if self.prev_keypoints is not None and i < len(self.prev_keypoints):
                prev_person = self.prev_keypoints[i]
                wrist_velocity = np.linalg.norm(right_wrist[:2] - prev_person[10][:2]) / frame_time
                if wrist_velocity > 0.5:  # Arbitrary threshold for fast movement
                    self.violence_stats['rapid_movement'] += 1
                    violent_frames += 1

            # 5. Proximity to others
            for j, other_person in enumerate(current_keypoints):
                if i != j and np.max(other_person[:, 2]) > self.confidence_threshold:
                    other_nose = np.squeeze(other_person)[0]
                    distance = np.linalg.norm(nose[:2] - other_nose[:2])
                    if distance < 0.1:  # Close proximity
                        self.violence_stats['close_proximity'] += 1
                        violent_frames += 1
                        break

        self.prev_keypoints = current_keypoints
        if violent_frames > 0:
            self.violence_detected = True
            return True
        return False

    def analyze_pose(self, keypoints):
        shaped = np.squeeze(keypoints)
        nose = shaped[0]
        left_hip = shaped[11]
        right_hip = shaped[12]
        left_knee = shaped[13]
        right_knee = shaped[14]
        if nose[2] > self.confidence_threshold and left_hip[2] > self.confidence_threshold:
            hip_level = (left_hip[0] + right_hip[0]) / 2
            knee_level = (left_knee[0] + right_knee[0]) / 2 if left_knee[2] > self.confidence_threshold else hip_level
            if nose[0] < hip_level:
                if abs(knee_level - hip_level) < 0.15:
                    self.pose_stats['standing'] += 1
                else:
                    self.pose_stats['crouching'] += 1
            else:
                self.pose_stats['sitting_or_lying'] += 1

    def process_frame(self, frame, frame_time):
        img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), 384, 640)
        input_img = tf.cast(img, dtype=tf.int32)
        results = movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
        
        current_people = 0
        for person in keypoints_with_scores:
            confidence = np.max(person[:, 2])
            if confidence > self.confidence_threshold:
                current_people += 1
                self.draw_connections(frame, person)
                self.draw_keypoints(frame, person)
                self.analyze_pose(person)
        
        # Check for violence
        is_violent = self.analyze_violence(keypoints_with_scores, frame_time)
        if is_violent:
            cv2.putText(frame, "VIOLENCE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        self.frame_count += 1
        if current_people > 0:
            self.detection_frames += 1
            self.total_people_detected += current_people
            self.max_people = max(self.max_people, current_people)
        
        return frame

    def summarize_video(self, video_path=None):
        cap = cv2.VideoCapture(0 if video_path is None else video_path)
        if not cap.isOpened():
            return "Error: Could not open video source"

        self.start_time = time.time()
        prev_time = self.start_time
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            frame_time = current_time - prev_time if current_time > prev_time else 1/30  # Default to 30 FPS
            prev_time = current_time
            
            processed_frame = self.process_frame(frame, frame_time)
            cv2.imshow('MoveNet MultiPose Summary', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.total_duration = time.time() - self.start_time
        
        cap.release()
        cv2.destroyAllWindows()
        
        summary = {
            "Total Frames": self.frame_count,
            "Frames with Detections": self.detection_frames,
            "Max People Detected": self.max_people,
            "Average People per Frame": round(self.total_people_detected / self.frame_count, 2) if self.frame_count > 0 else 0,
            "Total Duration (s)": round(self.total_duration, 2),
            "Detection Percentage": round((self.detection_frames / self.frame_count) * 100, 2) if self.frame_count > 0 else 0,
            "Average FPS": round(self.frame_count / self.total_duration, 2) if self.total_duration > 0 else 0,
            "Pose Statistics": dict(self.pose_stats),
            "Violence Statistics": dict(self.violence_stats),
            "Violence Detected": self.violence_detected
        }
        return summary

    def print_summary(self, summary):
        print("\n=== Video Summary ===")
        print(f"Total Frames Processed: {summary['Total Frames']}")
        print(f"Frames with Detections: {summary['Frames with Detections']}")
        print(f"Detection Percentage: {summary['Detection Percentage']}%")
        print(f"Maximum People Detected at Once: {summary['Max People Detected']}")
        print(f"Average People per Frame: {summary['Average People per Frame']}")
        print(f"Total Duration: {summary['Total Duration (s)']} seconds")
        print(f"Average FPS: {summary['Average FPS']}")
        print("Pose Statistics:")
        for pose_type, count in summary['Pose Statistics'].items():
            print(f"  {pose_type}: {count} frames")
        print("Violence Statistics:")
        for action, count in summary['Violence Statistics'].items():
            print(f"  {action}: {count} frames")
        print(f"Violence Detected: {'Yes' if summary['Violence Detected'] else 'No'}")

def main():
    summarizer = MoveNetSummarizer(confidence_threshold=0.1)
    video_path = "data/video/jogging.mp4"  # Replace with your video file path
    summary = summarizer.summarize_video(video_path)
    
    if isinstance(summary, dict):
        summarizer.print_summary(summary)
    else:
        print(summary)

if __name__ == "__main__":
    main()