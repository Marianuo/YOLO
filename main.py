import cv2 # OpenCV library for video processing and drawing
from ultralytics import YOLO # Import YOLOv8 pose model
import numpy as np # For numerical calculations
import smtplib # For sending emails
import os # For file operations
import time # For time-based calculations
from email.message import EmailMessage # For creating email messages

# Email Configuration
EMAIL_SENDER = ""  #Sender email address
EMAIL_PASSWORD = ""  #Sender email app password
EMAIL_RECEIVER = ""  # Recipient email

# Send Email Function
def send_email_alert(timestamp, images):
    msg = EmailMessage()
    msg["Subject"] = "Violence Alert: Nursing Home Incident"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content(f"Violence detected at {timestamp}. Please review the attached images.")

    # Attach images to email
    for i, img_path in enumerate(images):
        with open(img_path, "rb") as img:
            msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename=f"frame_{i + 1}.jpg")

    # Connect to SMTP server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

    print(f"Email alert sent with {len(images)} images!")

# Keypoint Indexes for Right and Left arms
keypoint_pairs = {
    "Right Arm": [10, 8, 6],  # Wrist → Elbow → Shoulder
    "Left Arm": [9, 7, 5] # Left Wrist → Left Elbow → Left Shoulder
}

# Constants
violence_threshold = 9500  # Speed threshold (px/sec) for detecting violence
min_displacement_threshold = 4 # Minimum pixels of movement to be counted(We ignore movements below 4 pixels and apply a cooldown timer so alerts aren't triggered by normal gestures or motion blur.


cooldown_period = 5   # Time (seconds) between email alerts to avoid spamming
visual_alert_duration = 2  # seconds
fps_default = 25

# Global tracking variables (global state between frames)
prev_keypoints = None # Previous frame keypoints
prev_box_centers = {} # Previous bounding box centers per person
prev_speeds = {} # Previous speeds for smoothing
last_alert_time = 0 # Last time email was sent
detection_buffer = [] # Buffer of last detected frames
image_sequence_folder = "violence_alert_images" # Folder to save images
os.makedirs(image_sequence_folder, exist_ok=True) # Create folder if not exists

# Visual alert control
visual_alert_frame_counter = 0
last_detected_box = None


# Main Processing Function
def process_video(video_path, output_path):
    global prev_keypoints, last_alert_time, detection_buffer
    global visual_alert_frame_counter, last_detected_box

    model = YOLO('yolov8m-pose.pt') # Load YOLOv8 pose model
    cap = cv2.VideoCapture(video_path) # Open video file

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps_default # Use default if not detected
    frame_interval = 1 / fps # Time between frames
    visual_alert_frames = int(visual_alert_duration * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Create output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read() # Read next frame, ret= true or false if the read was successful
        if not ret:
            break # End of video

        # Perform pose detection
        results = model(frame, conf=0.3) #confidence for person detect

        # If keypoints detected
        if results[0].keypoints is not None and len(results[0].keypoints) > 0: #checks if the YOLOv8 model successfully detected at least one person with pose keypoints
            keypoints = results[0].keypoints[0].xy.numpy() # Convert to numpy

            if prev_keypoints is None:
                prev_keypoints = keypoints.copy() # Store first frame

            speeds = {} # Store current frame speeds
            person_id = 0 # ID for each detected person

            # Process both arms
            for arm_name, indices in keypoint_pairs.items():
                arm_point = prev_arm_point = None

                # Try to get valid keypoint (wrist, elbow, shoulder)
                for idx in indices:
                    if idx < len(keypoints) and not np.all(keypoints[idx] == 0):
                        arm_point = keypoints[idx][:2]
                        prev_arm_point = prev_keypoints[idx][:2]
                        break

                # Fallback to bounding box center if no keypoints
                if arm_point is None or prev_arm_point is None:  #whether we failed to get keypoints for the arm (Right or Left)
                    if results[0].boxes is not None and len(results[0].boxes) > person_id: #Makeing sure that there’s a bounding box for the current detected person.
                        box = results[0].boxes.xyxy[person_id].cpu().numpy() #Retrieves the bounding box coordinates for that person
                        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) #Calculates the center point of the bounding box (an estimate of person position)
                        if person_id not in prev_box_centers: #If this is the first time we've seen this person, initialize their previous center
                            prev_box_centers[person_id] = center
                        arm_point = center #Use the bounding box center as a substitute for the missing keypoint.
                        prev_arm_point = prev_box_centers[person_id]
                        prev_box_centers[person_id] = center

                if arm_point is None or prev_arm_point is None: #If we still don’t have valid data, skip this person for this frame.
                    continue

                # Calculate speed
                dx = arm_point[0] - prev_arm_point[0]
                dy = arm_point[1] - prev_arm_point[1]
                displacement = np.sqrt(dx**2 + dy**2)

                if displacement >= min_displacement_threshold:
                    raw_speed = displacement / frame_interval #Displacement = how much a keypoint moved, Frame interval = how much time passed between frames
                    speed = 0.8 * prev_speeds.get(arm_name, raw_speed) + 0.2 * raw_speed #The formula smooths the speed by combining 80% of the previous speed with 20% of the current one to reduce sudden jumps and noise.
                    prev_speeds[arm_name] = speed
                    speeds[arm_name] = speed
                    print(f"{arm_name}: Speed = {speed:.2f} px/sec")

                    # Violence detection
                    if speed > violence_threshold and time.time() - last_alert_time > cooldown_period:
                        print("VIOLENCE DETECTED!")
                        visual_alert_frame_counter = visual_alert_frames

                        # Draw red bounding box
                        if results[0].boxes is not None and len(results[0].boxes) > person_id:
                            last_detected_box = results[0].boxes.xyxy[person_id].cpu().numpy()

                        # Save 3 frames for email
                        detection_buffer.append(frame)
                        if len(detection_buffer) > 3:
                            detection_buffer.pop(0)

                        # Send email with 3 images
                        if len(detection_buffer) == 3:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            files = []
                            for i, img in enumerate(detection_buffer):
                                path = f"{image_sequence_folder}/frame_{i}.jpg"
                                cv2.imwrite(path, img)
                                files.append(path)
                            send_email_alert(timestamp, files)
                            last_alert_time = time.time()

                person_id += 1

            # Draw keypoints and skeleton
            keypoints = keypoints[0]
            for x, y in keypoints[:, :2]:
                if not np.any(x == 0) and not np.any(y == 0):
                    cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

            # Define skeleton connections
            connections = [
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 6), (11, 12)
            ]
            # Draw skeleton lines
            for start, end in connections:
                if start < len(keypoints) and end < len(keypoints):
                    x1, y1 = keypoints[start][:2]
                    x2, y2 = keypoints[end][:2]
                    if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            prev_keypoints = keypoints.copy()  # Store keypoints for next frame

        # Draw red box & text for alert duration
        if visual_alert_frame_counter > 0:
            if last_detected_box is not None:
                x1, y1, x2, y2 = last_detected_box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            cv2.putText(frame, "VIOLENT ACTION DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            visual_alert_frame_counter -= 1

        # Write frame to output video
        out.write(frame)
        # Show frame in window
        cv2.imshow('YOLO Pose Processing', frame)
        # Break loop if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")

# Run
if __name__ == "__main__":
    input_video = "D:/Desktop/aiproject/IMG_8637.mov"
    output_video = "D:/Downloads/outputwithposes.mov"
    process_video(input_video, output_video)