import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def process_frame(frame):
    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect poses
    results = pose.process(image)
    
    # Convert back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw pose landmarks if detected
    if results.pose_landmarks:
        # For multi-person detection, we need to check pose_world_landmarks
        # MediaPipe Pose typically processes one person at a time
        # To handle multiple people, we'll need to use the Holistic solution or multiple instances
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    
    return image

# For processing a video file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the result
        cv2.imshow('MediaPipe Pose', processed_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# For processing a single image
def process_image(image_path):
    image = cv2.imread(image_path)
    processed_image = process_frame(image)
    
    # Display the result
    cv2.imshow('MediaPipe Pose', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # For video (uncomment and replace with your video path)
    # video_path = "path/to/your/video.mp4"
    # process_video(video_path)
    
    # For image (uncomment and replace with your image path)
    image_path = "data/violence_dataset/violence/V_4.mp4_frame0.jpg"
    process_image(image_path)

    # For webcam
    # cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
            
    #     processed_frame = process_frame(frame)
    #     cv2.imshow('MediaPipe Pose - Webcam', processed_frame)
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # cap.release()
    cv2.destroyAllWindows()

# Clean up
pose.close()