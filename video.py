import cv2
import os
import numpy as np

# Input video file (must be in the same folder as the script)
video_filename = "input_video.webm"  # Change this to your actual WebM file name
video_path = os.path.join(os.getcwd(), video_filename)

# Output folder for frames
output_folder = os.path.join(os.getcwd(), "images_50")

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the WebM video file
cap = cv2.VideoCapture(video_path)

# Get total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Check if the video file is loaded properly
if total_frames == 0:
    print("Error: Could not read the video file. Check the file path and format.")
    cap.release()
    exit()

# If video has fewer than 50 frames, use all frames
if total_frames < 50:
    frame_indices = list(range(total_frames))  
else:
    # Pick 50 evenly spaced frames across the entire video
    frame_indices = np.linspace(0, total_frames - 1, 50, dtype=int)

saved_count = 0

for frame_num in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # Jump to the selected frame
    ret, frame = cap.read()

    if ret:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count+1:03d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved: {frame_filename}")
        saved_count += 1

cap.release()
print(f"✅ Frame extraction complete! {saved_count} frames saved in 'images_50' folder.")

