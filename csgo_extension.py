
"""
CS2/CSGO Real-time Object Detection and Auto-Aim System
=======================================================

This script performs real-time object detection on CS2/CSGO gameplay using YOLOv8.
It detects player bodies and heads, automatically aims at enemy heads, and displays
the detection results with FPS counter.

Requirements:
- Raw input: OFF
- Mouse acceleration: OFF
- Sensitivity: 3.85 (you can change this, this is not a magical number)
- Screen capture area: 640x480 (top-left corner)

Classes detected:
- ct_body: Counter-Terrorist body
- ct_head: Counter-Terrorist head  
- t_body: Terrorist body
- t_head: Terrorist head

Controls:
- 'q': Exit the program
"""

import time
import cv2
import numpy as np
import pyautogui
from mss import mss
from ultralytics import YOLO

# Configuration constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
MODEL_PATH = "yolov8_csgo_cs2_model.pt" # same model as /model-results/csgo_model1_25_epoch/weights/best.pt
WINDOW_NAME = "CS2/CSGO Object Detection"

# Detection confidence threshold, you can change this value to adjust detection sensitivity
CONFIDENCE_THRESHOLD = 0.2

# Class mapping for detected objects
CLASS_LABELS = {
    1: "ct_body",
    2: "ct_head", 
    3: "t_body",
    4: "t_head"
}

# Target classes for auto-aim (heads only)
TARGET_CLASSES = ["t_head", "ct_head"]

# FPS tracking variables
frame_count = 1
start_time = time.time()
current_fps = 0

# Initialize YOLOv8 model
detection_model = YOLO(MODEL_PATH)

# Initialize screen capture
screen_capture = mss()

# Disable PyAutoGUI failsafe for competitive gaming
# pyautogui.FAILSAFE = False

print("Starting CS2/CSGO Object Detection System...")
print(f"Confidence threshold set to: {CONFIDENCE_THRESHOLD}")
print("Press 'q' or 'space' to exit")

# Main detection and auto-aim loop
while True:
    # Capture screen region (top-left 640x480)
    screen_region = {
        "top": 0,
        "left": 0, 
        "width": SCREEN_WIDTH,
        "height": SCREEN_HEIGHT
    }
    
    # Get screenshot and convert to numpy array
    raw_screenshot = np.array(screen_capture.grab(screen_region))
    
    # Extract RGB channels and ensure memory layout is contiguous
    # This optimization significantly improves inference speed
    game_frame = raw_screenshot[:SCREEN_HEIGHT, :SCREEN_WIDTH, :3]
    game_frame = np.ascontiguousarray(game_frame, dtype=np.uint8)
    
    # Run YOLOv8 inference on the game frame
    detection_results = detection_model(game_frame)
    
    # Process detection results
    detected_classes = []
    first_target_processed = False
    
    for detection_result in detection_results:
        # Extract class IDs for all detected objects
        if detection_result.boxes is not None:
            for class_id in detection_result.boxes.cls:
                detected_classes.append(int(class_id))
            
            # Process bounding boxes for auto-aim targeting
            for box_index, bounding_box in enumerate(detection_result.boxes.xyxy):
                # Get confidence score for this detection
                confidence_score = float(detection_result.boxes.conf[box_index])
                
                # Skip detections below confidence threshold
                if confidence_score < CONFIDENCE_THRESHOLD:
                    continue
                
                # Only process the first valid target to avoid multiple rapid clicks
                if not first_target_processed:
                    detected_class_name = CLASS_LABELS[detected_classes[box_index]]
                    
                    # Check if detected object is a target (enemy head)
                    if detected_class_name in TARGET_CLASSES:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
                        
                        # Calculate center point for precise aiming
                        target_center_x = int((x1 + x2) / 2)
                        target_center_y = int((y1 + y2) / 2)
                        
                        # Perform auto-aim: move cursor to target and click
                        pyautogui.moveTo(target_center_x, target_center_y)
                        pyautogui.click(x1 + 5, y1)  # Slight offset click for better accuracy
                        
                        # Draw detection rectangle (red color)
                        cv2.rectangle(game_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Display FPS on frame
                        fps_text = f"FPS: {current_fps:.1f}"
                        cv2.putText(game_frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
                        
                        first_target_processed = True
    
    # Display the processed frame with detections
    cv2.imshow(WINDOW_NAME, game_frame)
    
    # Handle keyboard input for program control
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        print("Exiting CS2/CSGO Object Detection System...")
        break
    
    # Calculate and update FPS
    elapsed_time = time.time() - start_time
    current_fps = frame_count / elapsed_time

    
    frame_count += 1

# Cleanup resources
cv2.destroyAllWindows()
print("Program terminated successfully.")