import cv2
import numpy as np
import os

def process_maritime_frame(frame):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to reduce water noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection
    # Adjust 50 and 150 based on your lighting conditions
    edges = cv2.Canny(blurred, 85, 255)
    
    # 4. Thresholding to create a solid binary mask of obstacles
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    
    # 5. Combine (Optional: focuses edges where threshold is high)
    combined = cv2.bitwise_and(edges, thresh)
    
    return edges, thresh, combined

# Path to your dataset folder
dataset_path = "/home/kate/datasets/Maritime_Visual_Tracking_Dataset_MVTD/train/32-USV/"
# frames = sorted([f for f in os.listdir(dataset_path) if f.endswith(('.jpg'))])

frames = os.listdir(dataset_path)

for frame_name in frames:
    img_path = os.path.join(dataset_path, frame_name)
    frame = cv2.imread(img_path)
    
    if frame is None:
        continue

    edges, thresh, result = process_maritime_frame(frame)

    # Show the results
    cv2.imshow('Original', frame)
    cv2.imshow('Canny Edges', edges)
    cv2.imshow('Thresholding', thresh)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
